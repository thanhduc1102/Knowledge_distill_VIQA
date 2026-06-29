#!/usr/bin/env python3
"""Acquire a DocFinQA slice via STREAMING (full dataset = 123K words/doc, times out on
normal load). Streams N examples per split and caches them to local JSONL so the
long-document evaluation is reproducible without re-downloading.

DocFinQA (Reddy et al., ACL 2024, arXiv 2401.06915): FinQA questions augmented with the
full SEC 10-K (avg 123K words). Repo: kensho/DocFinQA.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from itertools import islice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kensho/DocFinQA")
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="outputs/data/docfinqa")
    args = ap.parse_args()
    from datasets import load_dataset
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        t0 = time.time()
        try:
            ds = load_dataset(args.repo, split=split, streaming=True)
        except Exception as e:
            print(f"[{split}] stream ERR: {str(e)[:160]}", flush=True)
            summary[split] = {"error": str(e)[:200]}
            continue
        fp = out / f"{split}.jsonl"
        n = 0
        cols = None
        with fp.open("w") as fh:
            for ex in islice(ds, args.n):
                if cols is None:
                    cols = list(ex.keys())
                # keep fields but truncate the giant context to keep file sane (store full too? no)
                rec = dict(ex)
                fh.write(json.dumps(rec, default=str) + "\n")
                n += 1
                if n % 50 == 0:
                    print(f"  [{split}] {n} ... {time.time()-t0:.0f}s", flush=True)
        summary[split] = {"n": n, "cols": cols, "seconds": round(time.time() - t0, 1), "file": str(fp)}
        print(f"[{split}] saved {n} -> {fp} (cols={cols}) {time.time()-t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps(summary)[:400], flush=True)


if __name__ == "__main__":
    sys.exit(main())
