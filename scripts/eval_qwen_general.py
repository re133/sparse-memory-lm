"""Step 3, "Schadet es?": general abilities of Qwen3.5-0.8B with and without the add-on.

  .venv-qwen/bin/python scripts/eval_qwen_general.py --out report/qwen/general_Q.json
  .venv-qwen/bin/python scripts/eval_qwen_general.py --addons runs/qwen/QT-s0/addons.pt --out report/qwen/general_QT.json

1. Standard test (lm-evaluation-harness 0.4.13, zero-shot, accuracy; acc_norm where the harness reports it):
   MMLU, ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande; --limit examples per task (the same first examples
   for every model; MMLU: per subject).
2. Chat: CHAT_PROMPTS (6 German, 6 English), Qwen chat template, non-thinking mode, greedy, 200 new tokens.
   scripts/blind_chat.py later pairs the answers of two models in random order for a blinded judgement.
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.atomic import write_json  # noqa: E402
from smlm.qwen_memory import AddOnConfig, attach  # noqa: E402

TASKS = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]
CHAT_PROMPTS = [
    "Erkläre in drei Sätzen, wie eine Wärmepumpe funktioniert.",
    "Wer war Johann Wolfgang von Goethe, und welche zwei Werke sind am bekanntesten?",
    "Ein Zug fährt um 14:20 Uhr ab und braucht 2 Stunden 55 Minuten. Wann kommt er an?",
    "Schreibe eine kurze, höfliche E-Mail, in der du einen Termin am Dienstag absagst.",
    "Was ist der Unterschied zwischen Wetter und Klima?",
    "Nenne drei wichtige Ereignisse des Jahres 2026.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain briefly.",
    "Summarize the causes of World War I in four bullet points.",
    "Translate into German: 'The meeting was postponed because of the storm.'",
    "What happened in the world in 2026? Name a few events you know about.",
]


def load(model_dir, addons):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if addons:
        ck = torch.load(addons, map_location="cpu", weights_only=True)
        attach(model, AddOnConfig(**ck["addon_cfg"]))
        model.addons.load_state_dict(ck["state_dict"])
        model.addons.eval()
    # lm-eval wraps every model call in torch.autocast(enabled=False); the fp32 add-ons need bf16 autocast (as in
    # training), so autocast is switched on inside forward - for Q, Q+T and Q+D alike, so all three see the same numerics
    orig = model.forward

    def forward_bf16(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return orig(*a, **k)

    model.forward = forward_bf16
    return model, tok


@torch.no_grad()
def chat(model, tok, max_new=200):
    out = []
    for p in CHAT_PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        out.append({"prompt": p, "answer": tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("QWEN_DIR", os.path.join(ROOT, "models", "Qwen3.5-0.8B")))
    ap.add_argument("--addons", default=None)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    model, tok = load(args.model_dir, args.addons)
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit, num_fewshot=0,
                              bootstrap_iters=1000, log_samples=False)
    keep = {}
    for task, r in res["results"].items():
        keep[task] = {k: v for k, v in r.items() if isinstance(v, (int, float)) and ("acc" in k)}
    out = {"model_dir": args.model_dir, "addons": args.addons, "limit": args.limit, "tasks": keep,
           "n_samples": res.get("n-samples"), "chat": chat(model, tok),
           "seconds": round(time.time() - t0, 1), "lm_eval_version": __import__("lm_eval").__version__}
    write_json(args.out, out, indent=1, ensure_ascii=False)
    for t in args.tasks.split(","):
        r = keep.get(t, {})
        print(t, {k: round(v, 4) for k, v in r.items()})


if __name__ == "__main__":
    main()
