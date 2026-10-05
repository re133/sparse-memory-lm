"""Step 3 fact test: cloze items (numbers, names, dates) cut automatically from Wikipedia articles.

  .venv-qwen/bin/python scripts/make_fact_cloze.py --data_dir data/qwen_wiki \
      --out data/qwen_fact_cloze.jsonl

  * train: 500 items from training articles (train_new; the table has read them) -> "Wissen eingepflanzt?"
  * heldout: 500 items from the held-out articles (val_new) -> control, the table cannot know these facts
One item per article (articles drawn at random by a fixed hash). Item = prompt + answer:
  prompt = article title + blank line + the sentence containing the fact, cut right before the fact
  answer = the fact: a date ("12 March 2024"), a year, another number (>= 2 digits), or a name (2-4 capitalised words)
Filters so that the answer cannot be read off the prompt: the answer does not occur in the prompt (case-insensitive),
the sentence prefix has >= 5 words, the fact is not at the start of the sentence; years 2025 / 2026 are excluded
(almost every new article mentions them, so they are guessable).
Mix per split: 40 % names, 30 % dates/years, 30 % other numbers.
Scoring (scripts/eval_fact_cloze.py): greedy continuation, exact match at the start (first attempt), followed by a
non-alphanumeric character.
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(rf"\b(\d{{1,2}} (?:{MONTHS}) \d{{4}}|(?:{MONTHS}) \d{{1,2}}, \d{{4}})\b")
YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d{2,}(?:\.\d+)?)(?![\w,]*\d)")
NAME = re.compile(r"\b([A-Z][a-z]+(?: (?:of |de |van |von |da |del )?[A-Z][a-z]+){1,3})\b(?!-)")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
STOP_NAMES = set(MONTHS.split("|")) | {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                                       "The", "In", "He", "She", "It", "They", "His", "Her", "This", "After", "During"}
SEED = "fact-cloze-2026"


def h(*parts):
    return int.from_bytes(hashlib.blake2b(("|".join(map(str, parts)) + SEED).encode(), digest_size=8).digest(), "big")


def articles(data_dir, split, tok, eot):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    pages = meta["splits"][split]["pages"]
    t = np.fromfile(os.path.join(data_dir, split + ".bin"), dtype=np.uint32)
    cuts = np.flatnonzero(t == eot)
    starts = np.r_[0, cuts[:-1] + 1]
    assert len(starts) == len(pages), (len(starts), len(pages))
    for (pid, title), s, e in zip(pages, starts, cuts):
        yield pid, title, (s, e), t


def candidates(title, text):
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    out = []
    for para in body.split("\n"):
        for sent in SENT.split(para):
            for rx_kind, rx in (("date", DATE), ("year", YEAR), ("number", NUMBER), ("name", NAME)):
                for m in rx.finditer(sent):
                    kind = rx_kind
                    ans, a = m.group(1), m.start(1)
                    prefix = sent[:a].rstrip()
                    if a == 0 or len(prefix.split()) < 5 or not sent[a - 1] == " ":
                        continue
                    if kind == "year" and (ans in ("2025", "2026") or DATE.search(sent[max(0, a - 15):a + 5])):
                        continue
                    if kind == "number":
                        if YEAR.fullmatch(ans) or len(ans.replace(",", "").split(".")[0]) < 2:
                            continue
                        after_month = re.search(rf"(?:{MONTHS}) $", sent[:a])
                        quantity = re.match(r" [a-z]", sent[m.end(1):m.end(1) + 2])
                        if not (after_month or quantity):        # day of a date, or a counted quantity only
                            continue
                        if after_month:
                            kind = "date"
                    if kind == "name" and sent[max(0, a - 1):a] == "-":
                        continue
                    if kind == "name" and (ans.split()[0] in STOP_NAMES or any(w in STOP_NAMES for w in ans.split())):
                        continue
                    prompt = f"{title}\n\n{prefix}"
                    if ans.lower() in prompt.lower() or ans.lower() in title.lower():
                        continue
                    out.append({"type": "date/year" if kind in ("date", "year") else kind, "prompt": prompt,
                                "answer": ans})
    return out


def build(data_dir, split, n, tok, eot):
    want = {"name": int(0.4 * n), "date/year": int(0.3 * n)}
    want["number"] = n - sum(want.values())
    order = sorted(articles(data_dir, split, tok, eot), key=lambda a: h(split, a[0]))
    items, got = [], {k: 0 for k in want}
    for pid, title, (s, e), t in order:
        if all(got[k] >= want[k] for k in want):
            break
        text = tok.decode(t[s:e].tolist())
        cands = [c for c in candidates(title, text) if got[c["type"]] < want[c["type"]]]
        if not cands:
            continue
        c = min(cands, key=lambda c: h(pid, c["answer"], c["prompt"]))
        got[c["type"]] += 1
        items.append({"id": f"{split}-{pid}", "split": "train" if split == "train_new" else "heldout",
                      "page_id": pid, "title": title, **c})
    return items, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("QWEN_DATA", os.path.join(ROOT, "data", "qwen_wiki")))
    ap.add_argument("--tokenizer", default=os.environ.get("QWEN_DIR", os.path.join(ROOT, "models", "Qwen3.5-0.8B")))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    all_items = []
    for split in ("train_new", "val_new"):
        items, got = build(args.data_dir, split, args.n, tok, eot)
        print(split, len(items), got)
        all_items += items
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
