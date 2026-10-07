"""Opt-in compilation of dense block work, with eager memory and KV-cache paths.

Usage: compile_dense(model); compile_dense(model, enabled=False) restores eager forwards.
"""
from types import MethodType

import torch
from torch._functorch import config as functorch_config


def _dense_body(block, x, pos0, eng_out):
    if eng_out is not None:
        x = x + eng_out
    x = x + block.attn(block.attn_norm(x), None, pos0)
    return x + block.ffn(block.ffn_norm(x))


def _memory_input(block, x, pos0, eng_out):
    if eng_out is not None:
        x = x + eng_out
    x = x + block.attn(block.attn_norm(x), None, pos0)
    return x, block.ffn_norm(x)


def _residual(x, update):
    return x + update


def _forward(block, x, kv_cache=None, pos0=0, eng_rows=None):
    state = block._dense_compile
    if kv_cache is not None:
        # Cache mutation and the PKM decode graph stay on their existing inference path.
        return state["eager"](x, kv_cache, pos0, eng_rows)
    eng_out = block.engram(x, eng_rows, None) if block.engram is not None else None
    # train.py runs backward outside autocast; AOTAutograd otherwise assumes it stays enabled.
    with functorch_config.patch(backward_pass_autocast="off"):
        if not block.is_memory:
            return state["body"](block, x, pos0, eng_out)
        x, normed = state["body"](block, x, pos0, eng_out)
        # No compiled graph includes PKM/Engram, their custom autograd, or row-store mutation.
        return state["residual"](x, block.ffn(normed))


def compile_dense(model, enabled=True, *, backend="inductor"):
    """Compile dense block regions in place, preserving parameter identities and state_dict keys.

    Compilation is lazy and specializes for train/eval, autocast and input shapes. KV-cache calls
    remain eager. Call after moving the model to its device and before warming up or measuring it.
    This wraps forward callables rather than modules, so optimizers and existing checkpoints work
    unchanged. Full-module pickling is not supported; save the usual state_dict instead.
    As in train.py, run backward outside autocast.
    """
    for block in model.layers:
        state = getattr(block, "_dense_compile", None)
        if not enabled:
            if state is not None:
                if state["had_forward"]:
                    block.forward = state["eager"]
                else:
                    del block.forward
                del block._dense_compile
            continue
        if state is not None:
            continue
        body = _memory_input if block.is_memory else _dense_body
        block._dense_compile = {
            "eager": block.forward,
            "had_forward": "forward" in block.__dict__,
            # SDPA requires a concrete bool for is_causal; specialize sequence lengths explicitly.
            "body": torch.compile(body, backend=backend, fullgraph=True, dynamic=False),
            "residual": (torch.compile(_residual, backend=backend, fullgraph=True, dynamic=False)
                         if block.is_memory else None),
        }
        block.forward = MethodType(_forward, block)
    return model
