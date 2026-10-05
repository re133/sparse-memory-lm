"""Step 3, chat part of "Schadet es?": pair the chat answers of Qwen alone and a variant, blinded.

  python scripts/blind_chat.py --a runs/qwen_cloud/Q/general_ac.json --b runs/qwen_cloud/QT/general.json \
      --out report/qwen/chat_blind_Q_vs_QT.md --key report/qwen/chat_blind_Q_vs_QT.key.json

Per question the two answers appear as "Antwort 1" / "Antwort 2" in a random order (fixed seed per question);
the key (which answer is which model) goes into a separate file that the judge does not open before judging.
"""
import argparse
import hashlib
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", default="blind-chat-2026")
    args = ap.parse_args()
    a = json.load(open(args.a))["chat"]
    b = json.load(open(args.b))["chat"]
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]
    lines = ["# Chat-Vergleich (verblindet)", "",
             "Für jede Frage: Welche Antwort ist besser? **1**, **2** oder **gleich**. Antworten wurden gierig erzeugt "
             "(höchstens 200 Tokens, deshalb manchmal abgeschnitten). Die Zuordnung steht in einer separaten Datei.", ""]
    key = {}
    for i, (x, y) in enumerate(zip(a, b), 1):
        flip = hashlib.sha256(f"{args.seed}:{i}".encode()).digest()[0] % 2 == 1
        first, second = (y, x) if flip else (x, y)
        key[i] = {"1": os.path.basename(os.path.dirname(args.b)) if flip else os.path.basename(os.path.dirname(args.a)),
                  "2": os.path.basename(os.path.dirname(args.a)) if flip else os.path.basename(os.path.dirname(args.b))}
        lines += [f"## Frage {i}: {x['prompt']}", "", "**Antwort 1:**", "", "> " + first["answer"].strip().replace("\n", "\n> "),
                  "", "**Antwort 2:**", "", "> " + second["answer"].strip().replace("\n", "\n> "), "",
                  "Urteil: ____", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    json.dump(key, open(args.key, "w"), indent=1)
    print(f"{len(a)} pairs -> {args.out} (key: {args.key})")


if __name__ == "__main__":
    main()
