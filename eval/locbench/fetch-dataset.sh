#!/usr/bin/env bash
# Fetch Loc-Bench V1 (LocAgent, ACL 2025) into eval/data/locbench/dataset.jsonl.
# 560 instances of real GitHub issues with function-level ground truth
# (edit_functions = the functions the real fix modified). Re-runnable; skips
# if the dataset is already present. Needs network + uv (pulls `datasets`
# into an ephemeral env, same pattern as bench/fetch-corpora.sh).
set -euo pipefail
cd "$(dirname "$0")"

OUT=../data/locbench/dataset.jsonl
if [ -s "$OUT" ]; then
  echo "already fetched: $OUT ($(wc -l < "$OUT" | tr -d ' ') rows)"
  exit 0
fi

uv run --with datasets python - <<'PY'
from datasets import load_dataset
import json, pathlib

ds = load_dataset("czlll/Loc-Bench_V1")
split = "test" if "test" in ds else list(ds.keys())[0]
print("splits:", list(ds.keys()), "| using:", split)
print("columns:", ds[split].column_names)

out = pathlib.Path("../data/locbench/dataset.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
n = 0
with out.open("w") as f:
    for r in ds[split]:
        row = {
            "instance_id": r["instance_id"],
            "repo": r["repo"],
            "base_commit": r["base_commit"],
            "problem_statement": r["problem_statement"],
            "edit_functions": r["edit_functions"],
            # field name drifted across dataset revisions; keep whichever exists
            "category": r.get("category") or r.get("categories"),
        }
        f.write(json.dumps(row) + "\n")
        n += 1
print(f"wrote {n} rows to {out}")
print("sample:", json.dumps({k: row[k] for k in ("instance_id", "repo", "edit_functions", "category")}))
PY
