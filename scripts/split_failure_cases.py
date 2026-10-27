#!/usr/bin/env python3
"""
Split a failure-dataset run's cases into a demo half and a held-out half.

The held-out cases are never demonstrated — the VR app hides them, and
scripts/teach_failure_cases.py skips them — so evaluating on them measures
whether demos on some failures fix *other* failures, not just the demonstrated
ones. The split is stratified by object family (instance variants pooled), so
every held-out family also has demonstrated cases.

    python scripts/split_failure_cases.py data/vr_failures/<run>     # → <run>/split.json

split.json: {"seed", "heldout_frac", "demo": [case ids], "heldout": [case ids]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SPLIT_FILE = "split.json"


def family_of(shape: str) -> str:
    return shape.split(".v")[0]


def split_cases(cases: dict[str, str], heldout_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """cases: {case_id: shape} → (demo ids, held-out ids), stratified by family.
    A family with a single case goes to the demo half."""
    rng = random.Random(seed)
    by_family: dict[str, list[str]] = {}
    for case_id, shape in sorted(cases.items()):
        by_family.setdefault(family_of(shape), []).append(case_id)
    demo, heldout = [], []
    for ids in by_family.values():
        rng.shuffle(ids)
        k = int(len(ids) * heldout_frac) if len(ids) > 1 else 0
        heldout += ids[:k]
        demo += ids[k:]
    return sorted(demo), sorted(heldout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--heldout_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = p.parse_args()

    run = Path(args.run_dir)
    out = run / SPLIT_FILE
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (a split must not change once demos are recorded); --force to redo")
    cases = {f.stem: json.loads(f.read_text())["object"]["shape"] for f in sorted((run / "cases").glob("*.json"))}
    demo, heldout = split_cases(cases, args.heldout_frac, args.seed)
    out.write_text(json.dumps({"seed": args.seed, "heldout_frac": args.heldout_frac,
                               "demo": demo, "heldout": heldout}, indent=1))
    print(f"[split] {len(cases)} cases → {len(demo)} demo, {len(heldout)} held out → {out}")


if __name__ == "__main__":
    main()
