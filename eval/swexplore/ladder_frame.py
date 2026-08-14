#!/usr/bin/env python3
"""Write bench-ladder.jsonl: the registered rung order for the §27 campaign.

`eval_runner.py` applies `records[:limit]` **before** its resume filter, so
`--limit N --resume` is a growing *prefix*, not a chunk of N new. That is not a
limitation here, it is the mechanism: order the bench rows once, and every rung
is a longer prefix of the same file. No new runner flag, no chunk bookkeeping,
and nothing already paid for is re-paid.

Two properties the order must have, and both are asserted at the end:

1. **The pilot goes first.** Its 31 instances are already run, so putting them
   at the head makes every rung a superset and keeps R0 poolable.
2. **Every prefix is population-representative by language**, so R1 is not
   accidentally a Go campaign. Languages are merged by proportional stride
   rather than concatenated.

Within a language, ordering is `tierframe.interleaved_by_repo` — shuffle within
repo, shuffle the repo order, round-robin. That function is reused rather than
reimplemented because it carries the §18.6 fix: sorting *after* a shuffle is
stable and quietly undoes it, which made seeds 1 and 2 share 37 of 40 instances.

    python3 eval/swexplore/ladder_frame.py            # write it
    python3 eval/swexplore/ladder_frame.py --check    # verify an existing one
"""

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
sys.path.insert(0, str(HERE.parent / "locbench"))
from tierframe import interleaved_by_repo  # noqa: E402

SEED = 27
RUNGS = (31, 150, 848)


def proportional_merge(groups):
    """Merge per-language orders so any prefix keeps population proportions.

    Each language advances at a stride set by its share, so after k draws every
    language has contributed about `k * share` instances. Plain round-robin
    would over-represent the small languages in every prefix — C++ has one
    instance and would land in the first ten.
    """
    total = sum(len(g) for g in groups.values())
    # position = (taken + 0.5) / share -> the classic largest-remainder /
    # Sainte-Lague ordering. Lowest key goes next.
    heap = []
    for lang, g in groups.items():
        share = len(g) / total
        heap.append([0.5 / share, lang, 0, share])
    out = []
    while len(out) < total:
        heap.sort()
        pos, lang, taken, share = heap[0]
        out.append(groups[lang][taken])
        taken += 1
        if taken >= len(groups[lang]):
            heap.pop(0)
        else:
            heap[0] = [(taken + 0.5) / share, lang, taken, share]
    return out


def build():
    side = {}
    for line in (DATA / "sidecar.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            side[r["instance_id"]] = r
    pilot = json.loads((DATA / "pilot-frame.json").read_text())["instances"]
    pilot = [i for i in pilot if i in side]

    rest = [i for i in side if i not in set(pilot)]
    rng = random.Random(SEED)
    by_lang = collections.defaultdict(list)
    for i in rest:
        by_lang[side[i]["language"]].append(i)
    repo_of = {i: side[i]["repo"] for i in side}
    ordered = {lang: interleaved_by_repo(ids, repo_of, rng)
               for lang, ids in sorted(by_lang.items())}
    order = pilot + proportional_merge(ordered)
    assert len(order) == len(side) == len(set(order)), "order must be a permutation"
    return order, side


def write(order, side):
    bench = {}
    for line in (DATA / "bench.jsonl").read_text().splitlines():
        if line.strip():
            bench[json.loads(line)["instance_id"]] = line
    out = DATA / "bench-ladder.jsonl"
    out.write_text("".join(bench[i] + "\n" for i in order))
    meta = {
        "seed": SEED,
        "n": len(order),
        "rungs": list(RUNGS),
        "pilot_first": True,
        "source": "bench.jsonl + sidecar.jsonl + pilot-frame.json",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
        "order": order,
    }
    (DATA / "ladder-frame.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out, meta


# Distinct short labels: Java and JavaScript both truncate to "Java", which
# made the first frame report show two identical columns.
ABBR = {"Python": "py", "Go": "go", "JavaScript": "js", "TypeScript": "ts",
        "Rust": "rs", "Java": "java", "PHP": "php", "Ruby": "rb",
        "C": "c", "C++": "cpp"}


def report(order, side):
    pilot = set(json.loads((DATA / "pilot-frame.json").read_text())["instances"])
    pop = collections.Counter(side[i]["language"] for i in order)
    print(f"{'rung':>6} {'n':>5}  " + " ".join(f"{ABBR.get(l, l[:4]):>5}" for l in sorted(pop)))
    print(f"{'pop':>6} {len(order):>5}  " +
          " ".join(f"{pop[l] / len(order):5.0%}" for l in sorted(pop)))
    ok = True
    for n in RUNGS:
        pre = order[:n]
        c = collections.Counter(side[i]["language"] for i in pre)
        print(f"{n:>6} {n:>5}  " + " ".join(f"{c[l] / n:5.0%}" for l in sorted(pop)))
    # 1. prefix property: every rung contains every smaller rung, and R0 is the pilot
    for a, b in zip(RUNGS, RUNGS[1:]):
        if not set(order[:a]) <= set(order[:b]):
            print(f"FAIL: rung {a} is not contained in rung {b}"); ok = False
    if set(order[:RUNGS[0]]) != pilot:
        print(f"FAIL: first {RUNGS[0]} are not exactly the pilot frame"); ok = False
    else:
        print(f"\nok  first {RUNGS[0]} == pilot frame exactly (nothing re-paid)")
    # 2. every prefix beyond the pilot is within 8pp of population on Python,
    #    the only language big enough for a share to be meaningful
    for n in RUNGS[1:]:
        c = collections.Counter(side[i]["language"] for i in order[:n])
        drift = abs(c["Python"] / n - pop["Python"] / len(order))
        flag = "ok " if drift <= 0.08 else "FAIL"
        if drift > 0.08: ok = False
        print(f"{flag} rung {n}: Python share drifts {drift:+.1%} from population")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk frame instead of rewriting it")
    args = ap.parse_args()
    order, side = build()
    if args.check:
        have = json.loads((DATA / "ladder-frame.json").read_text())
        if have["order"] != order:
            print("FAIL: on-disk ladder order does not reproduce from seed", SEED)
            sys.exit(1)
        print(f"ok  ladder-frame.json reproduces from seed {SEED}")
    else:
        out, meta = write(order, side)
        print(f"wrote {out} ({meta['n']} rows, sha256={meta['sha256']})")
    sys.exit(0 if report(order, side) else 1)
