"""Chunked tied output projection and fp32 cross-entropy without retained vocabulary logits.

Usage: loss = chunked_cross_entropy(hidden, model.lm_head.weight, targets, chunk_size=1024)
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, reduction, ignore_index):
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size, ctx.reduction, ctx.ignore_index = chunk_size, reduction, ignore_index
        ctx.device_type = hidden.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        x, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, x.shape[0], chunk_size):
            logits = F.linear(x[start:start + chunk_size], weight)
            loss += F.cross_entropy(logits.float(), y[start:start + chunk_size],
                                    reduction="sum", ignore_index=ignore_index)
            del logits
        if reduction == "mean":
            loss /= (y != ignore_index).sum()
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        hidden, weight, targets = ctx.saved_tensors
        x, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        need_x, need_w = ctx.needs_input_grad[:2]
        grad_x = torch.empty_like(x) if need_x else None
        grad_w = torch.zeros_like(weight) if need_w else None
        scale = grad_loss
        if ctx.reduction == "mean":
            # Native mean CE has a NaN loss but zero gradients when every label is ignored.
            scale = scale / (y != ctx.ignore_index).sum().clamp_min(1)
        w = weight.detach().requires_grad_(need_w)
        for start in range(0, x.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            h = x[start:end].detach().requires_grad_(need_x)
            inputs = ([h] if need_x else []) + ([w] if need_w else [])
            # Rebuild only this chunk. Disabling the cast cache avoids sharing a freed cast graph
            # across chunks and keeps backward independent of the caller's current autocast state.
            with torch.enable_grad(), torch.autocast(ctx.device_type, dtype=ctx.autocast_dtype,
                                                     enabled=ctx.autocast_enabled, cache_enabled=False):
                logits = F.linear(h, w)
                loss = F.cross_entropy(logits.float(), y[start:end], reduction="sum",
                                       ignore_index=ctx.ignore_index)
            grads = torch.autograd.grad(loss, inputs, grad_outputs=scale)
            if need_x:
                grad_x[start:end].copy_(grads[0])
            if need_w:
                grad_w.add_(grads[-1])
            del logits, loss, grads
        return (grad_x.reshape(hidden.shape) if need_x else None), grad_w, None, None, None, None


def chunked_cross_entropy(hidden, weight, targets, chunk_size=1024, reduction="mean", ignore_index=-100):
    """Project (..., D) hidden states with (vocab, D) weights and reduce CE across token chunks.

    The caller's autocast controls the projection; CE and its reduction use fp32, matching the model.
    Backward recomputes each chunk, so saved tensors never include an (all tokens, vocab) allocation.
    Chunked reductions and bf16 weight-gradient matmuls change rounding; equivalence is numerical,
    not bitwise. Supports first derivatives, mean/sum reduction, and integer class labels only.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("expected hidden (..., D) and weight (vocab, D)")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match the hidden token dimensions")
    return _ChunkedCrossEntropy.apply(hidden, weight, targets, chunk_size, reduction, ignore_index)
