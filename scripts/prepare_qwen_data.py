"""Step 3 data: Wikipedia articles Qwen3.5 cannot know (created after its training) and articles it surely knows,
tokenised with Qwen's tokenizer.

  .venv-qwen/bin/python scripts/prepare_qwen_data.py --dump_dir /home/leon/smlm-data/enwiki-20260901 \
      --out /home/leon/smlm-data/qwen_wiki

Sources
  * new: English Wikipedia dump 20260901 (dumps.wikimedia.org), only the last part files of
    pages-articles-multistream27 (page ids >= 77,475,910; page ids grow with the creation time of a page).
    Main namespace, no redirects, no disambiguation pages, plain text >= 300 characters.
    Creation month of a page = from page-id thresholds read once from the page-creation log of the Wikipedia API
    (median page id of the first 50 creations of each month; cached in --out/page_id_months.json).
  * known: the 1,917 validation articles of stage 1b/1c (wikimedia/wikipedia 20231101.en, same selection as
    scripts/prepare_wikipedia.py). Qwen3.5 (released 2026-03-02) has almost certainly seen these texts.
Wikitext -> plain text: mwparserfromhell; templates, <ref>/<gallery>/tables, File/Image/Category links and the
sections "References", "External links", "See also", "Notes", "Further reading", "Sources", "Bibliography" are dropped.

Splits (deterministic hash of the page id)
  curve_YYYY-MM    up to 150 articles per creation month 2025-01 ... 2026-08: Qwen's PPL by month shows where its
                   knowledge ends (evaluation only, never trained)
  train_new        articles created on or after --train_from (default 2026-03), minus curve and val
  val_new          5 % of those (at most 1,500 articles): held out, the decisive "new knowledge" set
  mem_probe        1,000 articles of train_new (also trained on): how much the table stores
  val_known        the 2023 validation articles
Document = title + "\\n\\n" + text + <|endoftext|>; files are uint32 token ids; meta.json lists counts, thresholds and
the page ids / titles of every split.
"""
import argparse
import bz2
import glob
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.textprep import assign_split, unit_hash  # noqa: E402

SEED = 20260901
MONTHS = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)][:22]       # 2025-01 ... 2026-10
DROP_SECTIONS = {"references", "external links", "see also", "notes", "further reading", "sources",
                 "bibliography", "citations", "footnotes", "notes and references", "references and notes"}
UA = "smlm-research/0.1 (you@example.com)"


def month_thresholds(cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    for m in MONTHS:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=logevents&letype=create&lenamespace=0"
               f"&lestart={m}-01T00:00:00Z&ledir=newer&lelimit=50&format=json")
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60))
        ids = sorted(e["pageid"] for e in d["query"]["logevents"] if e.get("pageid"))
        out[m] = int(np.median(ids))
        time.sleep(0.5)
    out["_note"] = ("median page id of the first 50 main-namespace page creations of each month (Wikipedia API "
                    f"logevents, read {time.strftime('%Y-%m-%d')})")
    json.dump(out, open(cache, "w"), indent=1)
    return out


def month_of(pid, thr):
    best = None
    for m in MONTHS:
        if pid >= thr[m]:
            best = m
    return best


def clean(wikitext):
    import mwparserfromhell as mw
    code = mw.parse(wikitext)
    for sec in code.get_sections(levels=[2]):
        heads = sec.filter_headings(recursive=False)
        if heads and heads[0].title.strip_code().strip().lower() in DROP_SECTIONS:
            try:
                code.remove(sec)
            except ValueError:
                pass
    for t in code.filter_templates(recursive=False):
        try:
            code.remove(t)
        except ValueError:
            pass
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() in ("ref", "gallery", "table", "references", "math", "timeline", "imagemap",
                                    "score", "graph", "mapframe"):
            try:
                code.remove(tag)
            except ValueError:
                pass
    for link in code.filter_wikilinks(recursive=True):
        if re.match(r"\s*:?\s*(file|image|category)\s*:", str(link.title), re.I):
            try:
                code.remove(link)
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_part(args):
    path, min_pid = args
    out = []
    ns_tag = None
    with bz2.open(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag != "page":
                continue
            if ns_tag is None:
                ns_tag = el.tag[:-4]
            ns = el.findtext(ns_tag + "ns")
            pid = int(el.findtext(ns_tag + "id"))
            if ns == "0" and pid >= min_pid and el.find(ns_tag + "redirect") is None:
                title = el.findtext(ns_tag + "title")
                wt = el.findtext(f"{ns_tag}revision/{ns_tag}text") or ""
                low = wt.lower()
                if not ("(disambiguation)" in title or "{{disambig" in low or "{{dab" in low or "{{hndis" in low
                        or "{{geodis" in low or "{{set index" in low):
                    out.append((pid, title, wt))
            el.clear()
    return out


def clean_one(item):
    pid, title, wt = item
    try:
        text = clean(wt)
    except Exception:
        text = ""
    return (pid, title, text) if len(text) >= 300 else None


def known_articles():
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw_wikipedia", "*.parquet")))
    assert len(files) == 41
    out = []
    for f in files:
        tab = pq.read_table(f, columns=["id", "title", "text"])
        for aid, title, text in zip(*(tab.column(c).to_pylist() for c in ("id", "title", "text"))):
            if assign_split(unit_hash(aid, 20231101), 0.0003, 0.125) == "validation":
                out.append((aid, title, text))
    return out


def add_known_same(args):
    from transformers import AutoTokenizer
    raw = parse_part((args.known_same_part, 0))
    raw = sorted(raw, key=lambda r: unit_hash(r[0], SEED))[:6000]        # random subset before the slow cleaning
    with Pool(min(30, os.cpu_count())) as pool:
        pages = [p for p in pool.imap(clean_one, raw, chunksize=16) if p is not None][:1500]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    docs = [f"{t}\n\n{x}" for _, t, x in pages]
    ids = tok(docs, add_special_tokens=False)["input_ids"]
    arr = np.fromiter((i for d in ids for i in d + [eot]), dtype=np.uint32)
    arr.tofile(os.path.join(args.out, "val_known_same.bin"))
    meta_path = os.path.join(args.out, "meta.json")
    meta = json.load(open(meta_path))
    meta["splits"]["val_known_same"] = {"n_articles": len(pages), "n_tokens": int(arr.size),
                                        "n_words": int(sum(len(d.split()) for d in docs)),
                                        "source": os.path.basename(args.known_same_part),
                                        "pages": [[int(p), t] for p, t, _ in pages]}
    json.dump(meta, open(meta_path, "w"))
    print(f"val_known_same: {len(pages)} articles, {arr.size / 1e6:.2f} M tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="/home/leon/smlm-models/Qwen3.5-0.8B")
    ap.add_argument("--train_from", default="2026-03")
    ap.add_argument("--curve_per_month", type=int, default=150)
    ap.add_argument("--known_same_part", default=None,
                    help="only add val_known_same: 1,500 hash-sampled articles of this (old) dump part, cleaned exactly "
                         "like the new articles (control for the different text preparation of val_known)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.known_same_part:
        add_known_same(args)
        return
    thr = month_thresholds(os.path.join(args.out, "page_id_months.json"))
    parts = sorted(glob.glob(os.path.join(args.dump_dir, "*multistream27.xml-p*.bz2")))
    with Pool(len(parts)) as pool:
        raw = [p for part in pool.map(parse_part, [(p, thr["2025-01"]) for p in parts]) for p in part]
    print(f"read {len(raw)} candidate pages in {time.time() - t0:.0f} s", flush=True)
    with Pool(min(30, os.cpu_count())) as pool:
        pages = [p for p in pool.imap(clean_one, raw, chunksize=64) if p is not None]
    del raw
    print(f"cleaned: {len(pages)} articles created >= 2025-01 in {time.time() - t0:.0f} s", flush=True)

    splits = {k: [] for k in ("train_new", "val_new", "mem_probe", "val_known")}
    curve = {}
    pages.sort()
    for pid, title, text in pages:
        m = month_of(pid, thr)
        u = unit_hash(pid, SEED)
        if m is None or m > "2026-08":
            continue
        if m < args.train_from:
            curve.setdefault(m, []).append((u, pid, title, text))
            continue
        if u < 0.05:
            splits["val_new"].append((u, pid, title, text))
        else:
            curve.setdefault(m, []).append((u, pid, title, text))     # curve sample taken below
    for m, arts in curve.items():
        arts.sort()
        keep = arts[:args.curve_per_month]
        curve[m] = keep
        if m >= args.train_from:
            splits["train_new"] += arts[args.curve_per_month:]
    splits["val_new"] = sorted(splits["val_new"])[:1500]
    splits["train_new"].sort()                                       # random order (by hash)
    splits["mem_probe"] = splits["train_new"][:1000]
    splits["val_known"] = [(0.0, aid, title, text) for aid, title, text in known_articles()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    meta = {"source": "enwiki-20260901 pages-articles-multistream27 parts p77475910..p84130778 (new); "
                      "wikimedia/wikipedia 20231101.en validation articles of stage 1b (known)",
            "tokenizer": args.tokenizer, "eot_id": eot, "vocab_size": len(tok), "dtype": "uint32",
            "doc_format": "title + '\\n\\n' + text + <|endoftext|>", "seed": SEED, "train_from": args.train_from,
            "page_id_months": thr, "splits": {}}

    def write(name, arts):
        docs = [f"{t}\n\n{x}" for _, _, t, x in arts]
        ids = tok(docs, add_special_tokens=False)["input_ids"]
        arr = np.fromiter((i for d in ids for i in d + [eot]), dtype=np.uint32)
        arr.tofile(os.path.join(args.out, name + ".bin"))
        meta["splits"][name] = {"n_articles": len(arts), "n_tokens": int(arr.size),
                                "n_words": int(sum(len(d.split()) for d in docs)),
                                "pages": [[int(p), t] for _, p, t, _ in arts]}
        print(f"{name}: {len(arts)} articles, {arr.size / 1e6:.2f} M tokens", flush=True)

    for name, arts in splits.items():
        write(name, arts)
    for m in sorted(curve):
        write(f"curve_{m}", curve[m])
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
