r"""Let B-16M write some text, with its 6.4B-parameter table on the SSD, in RAM or in VRAM.

  python scripts/demo_generate.py --download                       # get the files from Hugging Face (~3.7 GB)
  python scripts/demo_generate.py --table nvme --prompt "Hamburg\n\nHamburg is"
  python scripts/demo_generate.py --table vram -i                  # type your own prompts

B-16M is the 21M-parameter model from the README with the 16.8M-row table (33M parameters used per token). The
table is the 4-bit file from scripts/convert_table.py (validation PPL 19.98, against 19.96 with the fp32 table):
  nvme  memory-mapped from the SSD, the 30% most-read rows cached in RAM (~1 GB). ~0.5 GB of VRAM.
  ram   whole 4-bit table in RAM (3.2 GB), rows copied to the GPU per token.
  vram  whole 4-bit table in VRAM (3.3 GB), decode graphs on. The fastest one.
The model was trained on 500M tokens of English Wikipedia, every article as "Title\n\nText", so prompts work best
in that form: "Volcano\n\nA volcano is" (type \n for a line break). It writes fluent Wikipedia-style English, but
the facts are mostly made up, it's a small model. Generation stops at the end of an article, at --tokens or at the
1024-token context.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.offload import HostTable, MmapQ4Table, load_model  # noqa: E402

HF_REPO = "re133/sparse-memory-lm-B-16M"
FILES = ["rest.pt", "values_q4.bin", "scales_q4.bin", "hot_rows.npy", "meta.json"]
EOT = 50256


def download(dest):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub first")
    snapshot_download(HF_REPO, local_dir=dest, allow_patterns=FILES)


def load(tables, kind, cache_frac):
    meta = json.load(open(os.path.join(tables, "meta.json")))
    rows, dim = meta["rows"], meta["dim"]
    qfile, sfile = os.path.join(tables, "values_q4.bin"), os.path.join(tables, "scales_q4.bin")
    if kind == "nvme":
        hot = np.load(os.path.join(tables, "hot_rows.npy"), mmap_mode="r")
        n = int(cache_frac * rows)
        table = MmapQ4Table(qfile, sfile, rows, dim, hot_rows=np.asarray(hot[:n]), cache_rows=n)
    else:
        p = torch.from_numpy(np.fromfile(qfile, dtype=np.uint8)).view(rows, dim // 2)
        s = torch.from_numpy(np.fromfile(sfile, dtype=np.float16))
        table = (p.cuda(), s.cuda()) if kind == "vram" else HostTable(p, s)
    model = load_model(os.path.join(tables, "rest.pt"), table)
    if kind == "vram":
        model.set_memory_decode_graphs(True)          # not possible with the table outside the GPU
    return model


def sample(logits, temperature, top_k, gen):
    if temperature <= 0:
        return int(logits.argmax())
    v, i = (logits / temperature).topk(top_k)
    return int(i[torch.multinomial(v.softmax(-1), 1, generator=gen)])


@torch.no_grad()
def generate(model, ids, n_new, temperature=0.8, top_k=40, seed=0, on_token=None):
    """Batch-1 sampling with KV cache. Returns the new tokens and the seconds spent after the prompt pass."""
    n_new = min(n_new, model.cfg.max_seq_len - len(ids))
    assert n_new > 0, f"the prompt fills the whole {model.cfg.max_seq_len}-token context"
    gen = torch.Generator(device="cuda").manual_seed(seed)
    caches = [dict() for _ in range(model.cfg.n_layers)]
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(torch.tensor([ids], device="cuda"), kv_caches=caches, pos0=0)
        t0 = time.perf_counter()
        for i in range(n_new):
            tok = sample(logits[0, -1].float(), temperature, top_k, gen)
            if tok == EOT:
                break
            out.append(tok)
            if on_token:
                on_token(out)
            if i + 1 < n_new:
                logits = model(torch.tensor([[tok]], device="cuda"), kv_caches=caches, pos0=len(ids) + i)
    return out, time.perf_counter() - t0


class Printer:
    """Prints the text as it comes. Decodes everything so far each time, so characters made of several tokens
    come out whole."""

    def __init__(self, enc):
        self.enc, self.done = enc, 0

    def __call__(self, toks):
        text = self.enc.decode(toks)
        if text.endswith("�"):                   # half a character, wait for the next token
            return
        print(text[self.done:], end="", flush=True)
        self.done = len(text)


def ram():
    """Own memory of the process, plus the pages of mapped files (nvme: page cache, Linux can drop it any time)."""
    st = dict(line.split(":", 1) for line in open("/proc/self/status"))
    anon, file = (int(st[k].split()[0]) / 2**20 for k in ("RssAnon", "RssFile"))
    return f"{anon:.1f} GB RAM + {file:.1f} GB mapped files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M")))
    ap.add_argument("--download", action="store_true", help=f"download the files from {HF_REPO} into --tables")
    ap.add_argument("--table", default="nvme", choices=["nvme", "ram", "vram"])
    ap.add_argument("--cache_frac", type=float, default=0.3, help="nvme: share of the rows cached in RAM")
    ap.add_argument("--prompt", default="Isaac Newton\\n\\nSir Isaac Newton was")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.download:
        download(args.tables)
    if not os.path.exists(os.path.join(args.tables, "values_q4.bin")):
        sys.exit(f"no table files in {args.tables} (--download, or --tables <dir> / SMLM_TABLES)")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    torch.set_num_threads(min(16, os.cpu_count()))

    t0 = time.perf_counter()
    model = load(args.tables, args.table, args.cache_frac)
    torch.cuda.synchronize()
    print(f"loaded in {time.perf_counter() - t0:.1f} s, table in {args.table}: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GB VRAM, {ram()}")
    generate(model, enc.encode_ordinary("Warm-up\n\n"), 8)        # Triton compiles its kernels on first use

    prompts = iter(lambda: input("\nprompt (empty line to quit)> "), "") if args.interactive else [args.prompt]
    try:
        for k, prompt in enumerate(prompts):
            ids = enc.encode_ordinary(prompt.replace("\\n", "\n"))
            print("\n" + enc.decode(ids), end="", flush=True)
            out, sec = generate(model, ids, args.tokens, args.temperature, args.top_k, args.seed + k, Printer(enc))
            print(f"\n\n[{len(out)} tokens in {sec:.2f} s = {len(out) / sec:.0f} tok/s, table in {args.table}, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB VRAM, {ram()}]")
    except (EOFError, KeyboardInterrupt):
        print()
    if args.table == "nvme":
        print("(the first run after a reboot reads the table from the SSD; after that Linux keeps the pages it "
              "read in the page cache, so later runs are faster)")


if __name__ == "__main__":
    main()
