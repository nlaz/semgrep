#!/usr/bin/env bash
# Drive a two-arm campaign to completion in resumable chunks.
#
# One canonical run per output file, one model; each chunk is a --resume
# slice. Stops cleanly when the usage window dies (run.py exits 3) instead
# of burning error rows, and stops when the frame is complete.
#
#   eval/locbench/campaign.sh              # the §16.9 frame: rg vs desc-v5
#   CHUNK=40 eval/locbench/campaign.sh     # smaller slices
#
# The arms are parameters rather than literals so a second A/B does not need
# a forked copy of this loop. The §19 description A/B (desc-v7 vs desc-v5)
# is paired *within* semgrep, so it wants its own output file and no rg arm:
#
#   OUT=../data/locbench/results-desc-v7.jsonl \
#   CONDITIONS=desc-v5,desc-v7 LIMIT=200 eval/locbench/campaign.sh
#
# Defaults reproduce the §16.9 invocation exactly (560 × 2 arms = 1120).
set -uo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-../data/locbench/results-scale.jsonl}"
CONDITIONS="${CONDITIONS:-rg,desc-v5}"
MODEL="${MODEL:-sonnet}"
LIMIT="${LIMIT:-560}"
CHUNK="${CHUNK:-80}"
WORKERS="${WORKERS:-3}"
# One ok row per (instance, arm), so the frame is instances × arms unless the
# caller overrides it. Derived rather than typed twice: a TARGET that silently
# disagrees with CONDITIONS is a loop that never terminates.
NCOND=$(awk -F, '{print NF}' <<<"$CONDITIONS")
TARGET="${TARGET:-$((LIMIT * NCOND))}"
echo "=== campaign: $CONDITIONS × $LIMIT instances ($MODEL) -> $OUT [target $TARGET]"

# Gate: does the tool answer what agents actually type? The §16.10 campaign
# spent $361 with 47% of one arm's searches silently empty because nobody
# asked (RESEARCH.md §16.11). Seconds, no API calls, hard stop.
echo "=== preflight"
python3 preflight.py || {
  echo "preflight failed — refusing to start a campaign on a broken tool."
  exit 4
}

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

  python3 run.py --limit "$LIMIT" --conditions "$CONDITIONS" --model "$MODEL" \
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
