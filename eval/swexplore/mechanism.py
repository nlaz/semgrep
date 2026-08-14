#!/usr/bin/env python3
"""§28.2 — bucketed loss accounting for the sub-sg / sub-rg head-to-head.

    python3 eval/swexplore/mechanism.py --run-id s27

For every paired instance where one arm out-scored the other on
`hit_region_rate`, attribute the loser's missing score to a cause, region by
region, from the session's own shim log and captured search output:

    A/B  line precision   submitted a region in a gold file that misses the
                          gold lines — split at 32 lines, the chunk size,
                          because a miss inside one window is the window's
                          edge and a miss beyond it is the wrong part of the
                          file entirely
    C/D  noise submission submitted a non-gold file — split by whether the
                          tool's own output displayed that file (C, the tool
                          led the agent there) or not (D, the agent's guess)
    E    buried           a gold file the winner submitted was present in the
                          loser's search output and never submitted
    F    scoped away      that gold file never appeared in any output and
                          every query was file/dir-scoped somewhere else
    F2   rank miss        never appeared despite at least one repo-wide query
    G    never invoked    the session made no calls to its tool at all

An instance's lost score is distributed proportionally across the miss
categories found in it, so the buckets sum to the total gap by construction.
Run for BOTH directions, always: a bucket is only a finding about a tool if
the other tool does not lose the same way (§28.2 — "scoped away" is the
largest rg bucket, so it is an agent behavior, not an sg defect).

Surface-text caveat: "appeared in output" is a full-path substring match on
the captured stdout. It was a path-OR-basename match until §32.4 measured
the damage — `tests.py` matches any repo's test file — so the E/C splits in
§28.2, computed under the old rule, read as upper bounds on "the tool showed
it" and lower bounds on "the agent guessed".
"""

import argparse
import collections
import json
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"

EXTS = (".py", ".js", ".rb", ".c", ".h", ".y", ".l", ".go", ".java",
        ".ts", ".tsx", ".rs", ".php", ".cs", ".cpp", ".test")


def load(run_id, arm):
    out = {}
    for line in open(DATA / "results" / f"{run_id}-{arm}.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if ((r.get("agent") or {}).get("status") or "ok") == "ok":
            out[r["instance_id"]] = r
    return out


def load_gold():
    gold = {}
    for line in open(DATA / "bench.jsonl"):
        b = json.loads(line)
        gt = b["ground_truth"]
        gold[b["instance_id"]] = {
            "files": set(gt["read_core_files"]),
            "regions": [(r["path"], r["start"], r["end"])
                        for r in gt["read_core_regions"]],
        }
    return gold


def session(run_id, iid, arm, tool):
    """[(argv, path_scoped, stdout_text)] for every unblocked call."""
    d = DATA / "runs" / run_id / iid / arm
    calls = []
    log = d / "shim_log.jsonl"
    if not log.exists():
        return calls
    for line in open(log):
        e = json.loads(line)
        if e.get("tool") != tool or e.get("blocked"):
            continue
        pos = [a for a in e["argv"] if not a.startswith("-")]
        scoped = any("/" in a or a.endswith(EXTS) for a in pos)
        out = d / "searches" / e.get("stdout_file", "\0")
        txt = out.read_text(errors="replace") if out.exists() else ""
        calls.append((e["argv"], scoped, txt))
    return calls


def overlaps(p, regs):
    return any(not (p["end"] < s or p["start"] > e) for s, e in regs)


def account(run_id, ids, loser, winner, larm, ltool, delta, gold):
    """Bucket the loser's missing score on instances where it lost."""
    share = collections.Counter()
    inst = collections.defaultdict(set)
    regions = collections.Counter()
    for i in ids:
        g = gold[i]
        greg = collections.defaultdict(list)
        for p, s, e in g["regions"]:
            greg[p].append((s, e))
        calls = session(run_id, i, larm, ltool)
        alltxt = "\n".join(t for _, _, t in calls)
        any_wide = any(not scoped for _, scoped, _ in calls)
        submitted = loser[i].get("regions") or []
        sub_paths = {p["path"] for p in submitted}
        local = collections.Counter()
        if not calls:
            local["G never invoked"] += 1
        else:
            for p in submitted:
                if p["path"] in greg:
                    if overlaps(p, greg[p["path"]]):
                        continue
                    dist = min(min(abs(p["start"] - e), abs(s - p["end"]))
                               for s, e in greg[p["path"]])
                    local["A line precision <=32 (chunk edge)" if dist <= 32
                          else "B line precision >32 (wrong area)"] += 1
                else:
                    # Full path only. A basename fallback matched
                    # `tests.py` and `loader.py`-inside-`dataloader.py`
                    # against files the tool never displayed, inflating C at
                    # D's expense — measured in §32.4, where fixing the same
                    # bug in misswhy.py moved ~150 regions between buckets.
                    shown = p["path"] in alltxt
                    local["C noise (tool showed it)" if shown
                          else "D noise (agent's own)"] += 1
            win_files = {p["path"] for p in winner[i].get("regions") or []} & g["files"]
            for f in win_files - sub_paths:
                if f in alltxt:
                    local["E gold surfaced, not submitted"] += 1
                elif not any_wide:
                    local["F gold scoped away (file-scoped only)"] += 1
                else:
                    local["F2 gold rank miss (had repo-wide query)"] += 1
        tot = sum(local.values()) or 1
        for b, n in local.items():
            share[b] += abs(delta[i]) * n / tot
            inst[b].add(i)
            regions[b] += n
    return share, inst, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s27")
    ap.add_argument("--arms", default="sub-sg:sg,sub-rg:rg",
                    help="two arm:tool pairs, first is the arm under study")
    args = ap.parse_args()
    (a_arm, a_tool), (b_arm, b_tool) = \
        [p.split(":") for p in args.arms.split(",")]

    A, B = load(args.run_id, a_arm), load(args.run_id, b_arm)
    gold = load_gold()
    ids = sorted(set(A) & set(B))
    M = "hit_region_rate"
    delta = {i: (A[i]["metrics"][M] or 0) - (B[i]["metrics"][M] or 0)
             for i in ids}
    vals = list(delta.values())
    a_loss = [i for i in ids if delta[i] < 0]
    b_loss = [i for i in ids if delta[i] > 0]
    print(f"{args.run_id}: n={len(ids)} paired ok  "
          f"mean {a_arm}−{b_arm} = {st.mean(vals):+.4f} (sd {st.pstdev(vals):.3f})  "
          f"w/l {len(b_loss)}/{len(a_loss)}  ties {sum(1 for v in vals if v == 0)}")
    hit = lambda r: (r["metrics"][M] or 0) > 0
    both = sum(1 for i in ids if hit(A[i]) and hit(B[i]))
    print(f"binary any-gold-region: both arms hit on {both}/{len(ids)} — "
          f"the margin is rate, not discovery\n")

    for name, lose_ids, loser, winner, larm, ltool in (
            (a_arm, a_loss, A, B, a_arm, a_tool),
            (b_arm, b_loss, B, A, b_arm, b_tool)):
        share, inst, regions = account(args.run_id, lose_ids, loser, winner,
                                       larm, ltool, delta, gold)
        gap = sum(abs(delta[i]) for i in lose_ids)
        print(f"== {name} lost {gap:.2f} rate-points on {len(lose_ids)} instances")
        for b, v in share.most_common():
            print(f"  {b:42s} {v:6.2f}  {100 * v / gap:5.1f}%  "
                  f"inst={len(inst[b]):3d}  regions={regions[b]}")
        print()


if __name__ == "__main__":
    main()
