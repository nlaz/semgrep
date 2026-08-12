#!/usr/bin/env python3
"""Per-region root-cause census for missed gold regions in one arm.

    python3 eval/swexplore/misswhy.py --run-id s32 --arm sub-sg --tool sg

Where mechanism.py (§28.2) buckets the *relative* gap on instances one arm
lost to the other, this walks EVERY missed gold region in every ok session
and asks how far down the funnel it got:

    tool surfaced the gold lines themselves        -> engine did its whole job
    tool surfaced the gold file (other lines)      -> engine found it, display/agent lost it
    agent read the file / grep found it            -> discovery happened outside the tool
    file appeared nowhere                          -> discovery failed; split by whether
                                                      any repo-wide ranked query ran

crossed with what the agent then did (submitted wrong lines in the file,
spent all five slots on other gold, left slots empty, never called the tool).

One bucket per region, checked in this order:

    NO_TOOL_CALLS      session never invoked the tool at all
    WRONG_LINES_NEAR   a submission is in the gold file, within 32 lines
    WRONG_LINES_FAR    a submission is in the gold file, further than that
    CROWDED_OUT        all five slots submitted and every one hits some other
                       gold region — a K-budget loss, not a search loss
    SEEN_NOT_SUB       the tool displayed the gold file; agent submitted
                       elsewhere (flags say whether the gold LINES were shown,
                       and the best display rank)
    READ_NOT_SUB       tool never showed it, but the transcript proves the
                       agent saw the path (Read/grep/other output)
    NEVER_SURFACED_W   nowhere in the session, despite >=1 repo-wide ranked
                       query — the retrieval failure bucket; queries recorded
                       for offline replay
    NEVER_SURFACED_S   nowhere, and every ranked query was scoped elsewhere
    NEVER_SURFACED_F   nowhere, and every repo-wide ranked query was floored

Instance-rate points: each region is worth 1/n_regions of its instance, so
bucket points sum to (1 - mean hit_region_rate) * n_instances give or take
the harness's line-clamp edge cases, which are reported as OVERLAP_DRIFT.

Output: a table, plus --json for per-region rows (instance, region, bucket,
flags, ranked queries) that the replay step consumes.
"""

import argparse
import collections
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"

# argv flags whose VALUE follows as a separate token — skip both when
# deciding whether a call was scoped by positional path args.
VALUED_FLAGS = {"--chunking", "--min-score", "-k", "--path", "--window",
                "--context", "-C", "-A", "-B", "--max-count", "-m"}


def load_rows(run_id, arm):
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
        gold[b["instance_id"]] = [(r["path"], r["start"], r["end"])
                                  for r in gt["read_core_regions"]]
    return gold


PATHLINE = re.compile(r"^([^\s:][^:]*):(\d+):", re.M)


def tool_calls(cell, tool):
    """Per unblocked tool call: (argv, stdout_text)."""
    calls = []
    log = cell / "shim_log.jsonl"
    if not log.exists():
        return calls
    for line in open(log):
        e = json.loads(line)
        if e.get("tool") != tool or e.get("blocked"):
            continue
        f = cell / "searches" / (e.get("stdout_file") or "\0")
        calls.append((e["argv"], f.read_text(errors="replace") if f.exists() else ""))
    return calls


def sg_traces(cell):
    """[(query, root, floored, mode)] for primary searches, in call order."""
    out = []
    t = cell / "trace.jsonl"
    if not t.exists():
        return out
    for line in open(t):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") != "search" or e.get("phase") != "primary":
            continue
        inp = e.get("input") or {}
        res = e.get("results") or {}
        out.append((inp.get("query", ""), inp.get("root", "."),
                    bool(res.get("floored")), e.get("mode") or inp.get("mode", "")))
    return out


def parse_shown(txt):
    """{path: set(lines)} shown in one captured stdout."""
    shown = collections.defaultdict(set)
    for m in PATHLINE.finditer(txt):
        shown[m.group(1)].add(int(m.group(2)))
    return shown


def display_rank(txt, path):
    """1-based rank of `path` among the blank-line-separated hit groups of a
    semantic sg output; None if absent. Keyword output has no groups — every
    file gets rank by first-appearance order instead, which is close enough
    for 'was it near the top'."""
    rank, cur = 0, None
    for block in txt.split("\n\n"):
        files = {m.group(1) for m in PATHLINE.finditer(block)}
        if not files:
            continue
        rank += 1
        if path in files:
            return rank
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s32")
    ap.add_argument("--arm", default="sub-sg")
    ap.add_argument("--tool", default="sg")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = load_rows(args.run_id, args.arm)
    gold = load_gold()
    ids = sorted(set(rows) & set(gold))

    buckets = collections.Counter()      # region counts
    points = collections.Counter()       # instance-rate points
    inst = collections.defaultdict(set)
    flags_c = collections.Counter()
    out_rows = []

    for iid in ids:
        regs = gold[iid]
        preds = rows[iid].get("regions") or []
        cell = DATA / "runs" / args.run_id / iid / args.arm
        calls = tool_calls(cell, args.tool)
        greps = tool_calls(cell, "grep") if args.tool != "grep" else []
        agent_greps = False
        traces = sg_traces(cell) if args.tool == "sg" else []

        # every path:line the tool itself displayed, plus per-call text
        shown = collections.defaultdict(set)
        for _, txt in calls:
            for p, lines in parse_shown(txt).items():
                shown[p] |= lines
        grep_txt = "\n".join(t for _, t in greps)
        # anything the agent saw at all: transcript includes tool results
        tf = cell / "transcript.jsonl"
        transcript = tf.read_text(errors="replace") if tf.exists() else ""
        # the agent's own searches, from its Bash tool_use blocks — the shim
        # log also carries Claude Code's startup greps, which are not the
        # agent choosing grep (a false 'used_grep_instead' otherwise)
        if transcript:
            for tl in transcript.splitlines():
                if '"name": "Bash"' not in tl and '"name":"Bash"' not in tl:
                    continue
                try:
                    ev = json.loads(tl)
                except json.JSONDecodeError:
                    continue
                for blk in (ev.get("message") or {}).get("content") or []:
                    if blk.get("type") == "tool_use" and blk.get("name") == "Bash":
                        cmd = (blk.get("input") or {}).get("command", "")
                        if re.search(r"\b(grep|rg|egrep|ack)\b", cmd):
                            agent_greps = True

        # scope evidence (sg: authoritative from trace envelopes)
        if args.tool == "sg" and traces:
            wide_ranked = [q for q, root, fl, mode in traces
                           if root in (".", "") and not fl]
            wide_floored = [q for q, root, fl, mode in traces
                            if root in (".", "") and fl]
        else:
            wide_ranked, wide_floored = [], []
            for argv, _ in calls:
                pos, skip = [], False
                for a in argv:
                    if skip:
                        skip = False
                        continue
                    if a in VALUED_FLAGS:
                        skip = True
                    elif not a.startswith("-"):
                        pos.append(a)
                if not any("/" in a or "." in a.rsplit("/", 1)[-1]
                           for a in pos[1:]):  # pos[0] is the pattern
                    wide_ranked.append(pos[0] if pos else "")

        pred_hits = []
        for p in preds:
            pred_hits.append(any(p["path"] == gp and
                                 not (p["end"] < s or p["start"] > e)
                                 for gp, s, e in regs))

        n_missed = 0
        for gp, s, e in regs:
            hit = any(q["path"] == gp and not (q["end"] < s or q["start"] > e)
                      for q in preds)
            if hit:
                continue
            n_missed += 1
            fl = set()
            same_file = [q for q in preds if q["path"] == gp]
            gold_lines = set(range(s, e + 1))
            sg_showed_file = gp in shown
            sg_showed_lines = bool(shown.get(gp, set()) & gold_lines)
            # full relative path only — a basename fallback matched `tests.py`
            # and `loader.py` (inside `dataloader.py`) against files the agent
            # never saw, misfiling NEVER_SURFACED regions as READ_NOT_SUB
            in_grep = gp in grep_txt
            in_transcript = gp in transcript
            best_rank = None
            if sg_showed_file:
                ranks = [display_rank(t, gp) for _, t in calls]
                ranks = [r for r in ranks if r]
                best_rank = min(ranks) if ranks else None

            if not calls:
                b = "NO_TOOL_CALLS"
                # shim_log records Claude Code's own startup greps too;
                # telemetry.json's grep_calls counts only the agent's
                if agent_greps:
                    fl.add("used_grep_instead")
            elif same_file:
                dist = min(min(abs(q["start"] - e), abs(s - q["end"]))
                           for q in same_file)
                b = "WRONG_LINES_NEAR" if dist <= 32 else "WRONG_LINES_FAR"
                if sg_showed_lines:
                    fl.add("tool_showed_gold_lines")
            elif len(preds) >= 5 and all(pred_hits):
                b = "CROWDED_OUT"
                if sg_showed_file:
                    fl.add("tool_showed_file")
            elif sg_showed_file:
                b = "SEEN_NOT_SUB"
                if sg_showed_lines:
                    fl.add("tool_showed_gold_lines")
                if best_rank:
                    fl.add(f"rank<={1 if best_rank == 1 else (3 if best_rank <= 3 else 5)}"
                           if best_rank <= 5 else "rank>5")
                if in_transcript and f"Read" in transcript:
                    pass
            elif in_grep:
                b = "READ_NOT_SUB"
                fl.add("via_grep")
            elif in_transcript:
                b = "READ_NOT_SUB"
                fl.add("via_transcript_only")
            else:
                if wide_ranked:
                    b = "NEVER_SURFACED_W"
                elif wide_floored:
                    b = "NEVER_SURFACED_F"
                else:
                    b = "NEVER_SURFACED_S"
            if len(preds) < 5:
                fl.add("slots_unused")
            buckets[b] += 1
            points[b] += 1.0 / len(regs)
            inst[b].add(iid)
            for f in fl:
                flags_c[(b, f)] += 1
            out_rows.append({"instance_id": iid, "region": [gp, s, e],
                             "bucket": b, "flags": sorted(fl),
                             "best_rank": best_rank,
                             "n_regions": len(regs), "n_preds": len(preds),
                             "wide_queries": wide_ranked[:8]})

    tot_regions = sum(len(gold[i]) for i in ids)
    missed = sum(buckets.values())
    lost_points = sum(points.values())
    print(f"{args.run_id}/{args.arm}: {len(ids)} sessions, {tot_regions} gold regions, "
          f"{missed} missed ({missed / tot_regions:.1%})")
    print(f"instance-rate points lost: {lost_points:.1f} "
          f"(= {lost_points / len(ids):.4f} of the mean rate)\n")
    print(f"{'bucket':20s} {'regions':>7s} {'share':>6s} {'points':>7s} {'pts%':>6s} {'inst':>5s}")
    for b, n in buckets.most_common():
        print(f"{b:20s} {n:7d} {n / missed:6.1%} {points[b]:7.1f} "
              f"{points[b] / lost_points:6.1%} {len(inst[b]):5d}")
    print("\nflags within buckets:")
    for (b, f), n in sorted(flags_c.items(), key=lambda kv: -kv[1]):
        print(f"  {b:20s} {f:28s} {n}")

    if args.json:
        with open(args.json, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(out_rows)} region rows -> {args.json}")


if __name__ == "__main__":
    main()
