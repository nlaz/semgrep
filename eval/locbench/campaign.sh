#!/usr/bin/env bash
# Drive the §16.9 campaign to completion in resumable chunks.
#
# One canonical run (results-scale.jsonl, one model); each chunk is a
# --resume slice. Stops cleanly when the usage window dies (run.py exits 3)
# instead of burning error rows, and stops when the frame is complete.
#
#   eval/locbench/campaign.sh              # run until done or the window dies
#   CHUNK=40 eval/locbench/campaign.sh     # smaller slices
set -uo pipefail
cd "$(dirname "$0")"

OUT=../data/locbench/results-scale.jsonl
CHUNK="${CHUNK:-80}"
TARGET="${TARGET:-1120}"
WORKERS="${WORKERS:-3}"

while :; do
  ok=$(python3 - "$OUT" <<'PY'
import json, sys
seen = set()
try:
    for line in open(sys.argv[1]):
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("status") == "ok":
            seen.add((r["instance_id"], r["condition"], r.get("model")))
except FileNotFoundError:
    pass
print(len(seen))
PY
)
  echo "=== campaign: $ok/$TARGET ok rows ($(date +%H:%M:%S))"
  [ "$ok" -ge "$TARGET" ] && { echo "campaign complete"; break; }
  # Abandoned cells make TARGET unreachable, so "no progress last chunk" is
  # the real completion signal — without it the loop spins on an empty todo.
  if [ "${prev_ok:-}" = "$ok" ]; then
    echo "campaign complete: no progress last chunk ($ok rows; the rest are "
    echo "abandoned cells). Analyze with ab_analyze.py."
    break
  fi
  prev_ok=$ok

  python3 run.py --limit 560 --conditions rg,desc-v5 --model sonnet \
    --resume --max-new "$CHUNK" --evict-mirrors --workers "$WORKERS" --out "$OUT"
  rc=$?
  if [ "$rc" -eq 3 ]; then
    echo "=== usage window exhausted; stopping. Re-run this script when it resets."
    exit 3
  elif [ "$rc" -ne 0 ]; then
    echo "=== chunk failed with exit $rc; stopping."
    exit "$rc"
  fi
done
