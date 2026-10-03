"""Download the raw datasets at pinned revisions, rebuild the token files and check that everything is
byte-identical to the data used at home (cloud/data_sha256.txt).

  python cloud/fetch_data.py            (exit code 0 only if every hash matches)

  * WikiText-103 raw (Salesforce/wikitext, wikitext-103-raw-v1) -> data/raw -> scripts/prepare_data.py
    (needed for the secondary WikiText validation set and for excluding WikiText val/test articles)
  * English Wikipedia 20231101.en (wikimedia/wikipedia, 41 parquet files, ~11 GB) -> data/raw_wikipedia ->
    scripts/prepare_wikipedia.py (500 M training tokens + validation set of stage 1b / 1c)
"""
import concurrent.futures as cf
import hashlib
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SOURCES = [  # (HF dataset, pinned revision, path prefix in the repo, local directory, file names)
    ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-103-raw-v1", "raw",
     ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "validation-00000-of-00001.parquet",
      "test-00000-of-00001.parquet"]),
    ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en", "raw_wikipedia",
     [f"train-{i:05d}-of-00041.parquet" for i in range(41)]),
]


def expected():
    out = {}
    for line in open(os.path.join(ROOT, "cloud", "data_sha256.txt")):
        if line.strip() and not line.startswith("#"):
            h, p = line.split()
            out[p] = h
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, want):
    if os.path.exists(dest) and sha256(dest) == want:
        return dest, "cached"
    tmp = dest + ".part"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            if sha256(tmp) != want:
                raise ValueError("sha256 mismatch")
            os.replace(tmp, dest)
            return dest, "ok"
        except Exception as e:                          # noqa: BLE001
            print(f"  retry {attempt + 1} for {os.path.basename(dest)}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"download failed: {url}")


def main():
    exp = expected()
    t0 = time.time()
    jobs = []
    for repo, rev, prefix, local, files in SOURCES:
        os.makedirs(os.path.join(DATA, local), exist_ok=True)
        for f in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{prefix}/{f}"
            jobs.append((url, os.path.join(DATA, local, f), exp[f"{local}/{f}"]))
    with cf.ThreadPoolExecutor(6) as ex:
        for dest, status in ex.map(lambda j: download(*j), jobs):
            print(f"{status:6s} {os.path.relpath(dest, ROOT)}", flush=True)
    print(f"raw data verified ({len(jobs)} files, {time.time() - t0:.0f} s)", flush=True)
    for script in ("prepare_data.py", "prepare_wikipedia.py"):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script)], cwd=ROOT, check=True)
    bad = []
    for p, h in exp.items():
        if p.startswith("raw"):
            continue
        got = sha256(os.path.join(DATA, p))
        print(f"{'OK ' if got == h else 'BAD'} {p}", flush=True)
        if got != h:
            bad.append(p)
    if bad:
        raise SystemExit(f"prepared data differs from home: {bad}")
    print(f"all data byte-identical to home ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
