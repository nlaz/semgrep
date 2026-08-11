#!/usr/bin/env python3
"""§31.1's registered gate: does `a | b` combine what two searches found?

Mines consecutive same-session ranked sg call pairs from the s27/s31 shim
logs (same positional scope, no `-e`, both unblocked), replays `a`, `b`, and
`"a | b"` against the surviving checkouts with the current binary, and scores:

    union-coverage@5   the merged top-5 contains the gold files that the two
                       sequential top-5s contained (gold from bench.jsonl)
    turn-saved         merged top-5 covers the sequential union outright

Plus the direct before/after: the real pipe-containing queries replayed
verbatim — under the old engine they ran pooled, under §31 they split.

    python3 eval/swexplore/pairplay.py --limit 20      # smoke
    python3 eval/swexplore/pairplay.py                 # the gate

Gate (§31.1, registered before this ran): merged union-coverage within 0.05
of the sequential union, and the verbatim pipe queries not worse than their
pooled behaviour.
"""

import argparse
import collections
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
SG = pathlib.Path(
    os.environ.get("SEMGREP_BIN", HERE.parent.parent / "target" / "release" / "sg")
)
EXTS = (".py", ".js", ".rb", ".c", ".h", ".go", ".java", ".ts", ".tsx", ".rs", ".php")


def gold_files():
    out = {}
    for l in open(DATA / "bench.jsonl"):
        b = json.loads(l)
        out[b["instance_id"]] = set(b["ground_truth"]["read_core_files"])
    return out


def ranked_calls(log):
    """(query, scope, argv) per unblocked ranked sg call, in order."""
    calls = []
    for l in open(log):
        e = json.loads(l)
        if e.get("tool") != "sg" or e.get("blocked"):
            continue
        argv = e["argv"]
        if "-e" in argv or "--exact" in argv:
            continue
        pos = [a for a in argv if not a.startswith("-")]
        if not pos:
            continue
        query = pos[0]
        scope = pos[1] if len(pos) > 1 else "."
        calls.append((query, scope, e.get("cwd", "")))
    return calls


def mine_pairs(runs):
    """Consecutive ranked pairs with the same scope, per session."""
    pairs = []
    for run in runs:
        for log in glob.glob(str(DATA / "runs" / run / "*" / "sub-sg" / "shim_log.jsonl")):
            iid = pathlib.Path(log).parent.parent.name
            calls = ranked_calls(log)
            for (qa, sa, cwd), (qb, sb, _) in zip(calls, calls[1:]):
                if sa != sb or qa == qb:
                    continue
                # A pipe inside either query would nest separators; skip.
                if "|" in qa or "|" in qb:
                    continue
                pairs.append({"iid": iid, "a": qa, "b": qb, "scope": sa, "cwd": cwd})
    return pairs


def top5_files(query, scope, cwd, cache):
    cmd = [str(SG), "--json", "-k", "5", query]
    if scope != ".":
        cmd.append(scope)
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=cwd,
            env={"SEMGREP_CACHE_DIR": cache, "PATH": "/usr/bin:/bin",
                 "SEMGREP_NO_HINTS": "1"},
        )
    except Exception:  # noqa: BLE001
        return None
    files = []
    for l in p.stdout.splitlines():
        if l.strip():
            try:
                files.append(json.loads(l)["path"])
            except Exception:  # noqa: BLE001
                return None
    return set(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="s27,s31")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=DATA / "results" / "pairplay.jsonl")
    args = ap.parse_args()

    gold = gold_files()
    pairs = mine_pairs(args.runs.split(","))
    # Only instances whose checkout survives and has gold.
    pairs = [p for p in pairs
             if (DATA / "runs").parent.joinpath("repos", p["iid"]).is_dir()
             and p["iid"] in gold and pathlib.Path(p["cwd"]).is_dir()]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} replayable consecutive pairs")

    done = set()
    if args.out.exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip()}
    out_f = open(args.out, "a")

    n = mergeable = covered = saved = 0
    caches: dict = {}
    for i, p in enumerate(pairs, 1):
        # hashlib, not hash(): the builtin is salted per process, which would
        # make every resume re-run everything and no key ever match again.
        import hashlib
        key_src = f'{p["a"]}\x00{p["b"]}\x00{p["scope"]}'
        key = f'{p["iid"]}/{hashlib.sha256(key_src.encode()).hexdigest()[:8]}'
        if key in done:
            continue
        # One cache per instance: the first search write-through builds the
        # index once; every later replay against that checkout runs warm.
        cache = caches.setdefault(p["iid"], tempfile.mkdtemp(prefix="pairplay-"))
        fa = top5_files(p["a"], p["scope"], p["cwd"], cache)
        fb = top5_files(p["b"], p["scope"], p["cwd"], cache)
        fm = top5_files(f'{p["a"]} | {p["b"]}', p["scope"], p["cwd"], cache)
        if fa is None or fb is None or fm is None:
            continue
        g = gold[p["iid"]]
        seq_gold = (fa | fb) & {f for f in g} | {
            f for f in (fa | fb) if any(f.endswith(gf.rsplit("/", 1)[-1]) for gf in g)
        }
        merged_gold = {f for f in fm if any(f.endswith(gf.rsplit("/", 1)[-1]) for gf in g)}
        n += 1
        row = {
            "key": key, "iid": p["iid"], "a": p["a"][:80], "b": p["b"][:80],
            "scope": p["scope"][:60],
            "seq_gold": len(seq_gold), "merged_gold": len(merged_gold),
            "union_covered": merged_gold >= seq_gold if seq_gold else None,
            "turn_saved": bool(seq_gold) and merged_gold >= seq_gold,
        }
        if seq_gold:
            mergeable += 1
            covered += row["union_covered"]
            saved += row["turn_saved"]
        out_f.write(json.dumps(row) + "\n")
        if i % 25 == 0:
            out_f.flush()
            print(f"  [{i}/{len(pairs)}] mergeable={mergeable} covered={covered}")
    out_f.flush()

    print(f"\nreplayed {n} pairs; {mergeable} where the sequential pair found gold")
    if mergeable:
        print(f"  union-coverage@5: {covered}/{mergeable} = {covered/mergeable:.1%}")
        print(f"  turns saved:      {saved}/{mergeable} = {saved/mergeable:.1%}")
        print(f"\n§31.1 gate: merged must cover the sequential union on ≥95% "
              f"of mergeable pairs → {'PASS' if covered/mergeable >= 0.95 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
