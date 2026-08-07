#!/usr/bin/env python3
"""§25's endpoints: what a display format does to agent behaviour.

`ab_analyze.py` is built around McNemar over binary per-instance outcomes,
which is right for accuracy and wrong here. §25's primary endpoint is a
*count* — how many times an agent opened a file immediately after searching —
so the test is a paired bootstrap over instance-level differences and the
script is separate rather than bent into a shape it was not registered for.

Everything is read from the transcripts, which record every `tool_use` and its
`tool_result`. That is also where §25.1's power numbers came from: run this
against the existing campaigns and it reproduces the 1.98 baseline and the 1.47
paired sd the registration was sized on.

    python3 eval/locbench/displaycmp.py --results ../data/locbench/results-display.jsonl \\
        --arms disp-line,disp-full,disp-head,rg --base disp-line

    # reproduce §25.1's power inputs from what was already on disk
    python3 eval/locbench/displaycmp.py --baseline-only
"""

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from ab_analyze import boot_ci  # noqa: E402  — one bootstrap, one convention

DATA = HERE.parent / "data" / "locbench"
SEARCH_TOOLS = ("rg", "semgrep", "sg", "search")

# What a truncated tool result looks like from the agent's side. §25.1 tripwire
# 6: at 20x the bytes the treatment can backfire by having its own lower-ranked
# hits deleted, and that failure is invisible in every other endpoint.
TRUNC_MARKERS = ("[Result truncated", "tool result too large", "<response clipped>")


def walk(path):
    """-> (tool_use name, command, result_text) triples, in order."""
    pending, out = {}, []
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # `message` is usually a dict, and sometimes a bare string — an
        # interrupted run writes at least one such line, so a walker that
        # assumes the common shape dies on exactly the campaigns worth
        # analysing. Same for `content`, which can be a string.
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        blocks = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                cmd = (b.get("input") or {}).get("command", "") or ""
                pending[b.get("id")] = (b.get("name"), cmd)
                out.append([b.get("name"), cmd, ""])
            elif b.get("type") == "tool_result":
                # Attach to the most recent unresolved use of that id.
                name_cmd = pending.pop(b.get("tool_use_id"), None)
                if name_cmd is None:
                    continue
                text = b.get("content")
                text = text if isinstance(text, str) else json.dumps(text)
                for row in reversed(out):
                    if row[0] == name_cmd[0] and row[1] == name_cmd[1] and not row[2]:
                        row[2] = text
                        break
    return out


def is_search(name, cmd, tool):
    """A search by THIS arm's tool, not any tool.

    `tool` rather than a fixed list: an arm told to type `sg` that emits
    `semgrep` is escaping its own treatment, and counting it here would hide
    that in the endpoint instead of surfacing it.
    """
    return name == "Bash" and cmd.strip().startswith(tool + " ")


def metrics(path, tool):
    seq = walk(path)
    reads = sum(1 for n, _, _ in seq if n == "Read")
    searches = sum(1 for n, c, _ in seq if is_search(n, c, tool))
    after = 0
    for (n1, c1, _), (n2, _, _) in zip(seq, seq[1:]):
        if is_search(n1, c1, tool) and n2 == "Read":
            after += 1
    sbytes = [len(r) for n, c, r in seq if is_search(n, c, tool)]
    trunc = sum(1 for n, c, r in seq
                if is_search(n, c, tool) and any(m in r for m in TRUNC_MARKERS))
    return {
        "reads_after_search": after,
        "reads": reads,
        "searches": searches,
        "search_bytes": sum(sbytes),
        "median_search_bytes": st.median(sbytes) if sbytes else 0,
        "truncated_searches": trunc,
    }


def collect(runs_dir, tool_of, frame=None):
    """-> {(run, instance): {condition: metrics}} for transcripts on disk.

    `frame` restricts to a registered instance list. Without it the walk picks
    up every run ever made, including the smoke runs that sized the campaign —
    which are the same arms and the same binary, and still not the frame that
    was registered. Mixing them would quietly enlarge n with instances chosen
    after the fact.
    """
    out = defaultdict(dict)
    for fp in Path(runs_dir).glob("*/*/*/transcript.jsonl"):
        cond, inst, run = fp.parts[-2], fp.parts[-3], fp.parts[-4]
        if frame is not None and inst not in frame:
            continue
        out[(run, inst)][cond] = metrics(fp, tool_of(cond))
    return out


def paired(data, base, cand, key):
    return [(v[base][key], v[cand][key]) for v in data.values()
            if base in v and cand in v]


def line(label, pairs):
    if len(pairs) < 5:
        return f"  {label:<26} (n={len(pairs)}, too few paired runs)"
    a = st.mean(x for x, _ in pairs)
    b = st.mean(y for _, y in pairs)
    # `boot_ci` returns mean(first) - mean(second), so the pairs go in swapped:
    # every heading here reads "cand - base", and a Δ carrying the opposite
    # sign to its own label is how a reduction gets published as an increase.
    d, lo, hi = boot_ci([(y, x) for x, y in pairs])
    sd = st.pstdev([y - x for x, y in pairs])
    star = " " if lo <= 0 <= hi else "*"
    return (f"  {label:<26} n={len(pairs):>4}  base={a:>8.3f} cand={b:>8.3f}  "
            f"Δ={d:>+8.3f} [{lo:>+7.3f},{hi:>+7.3f}]{star}  sd(diff)={sd:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--arms", default="disp-line,disp-full,disp-head,rg")
    ap.add_argument("--base", default="disp-line")
    ap.add_argument("--frame", type=Path, default=None,
                    help="registered instance list (display-frame-280.json); "
                         "without it every run on disk is included, smoke "
                         "runs among them")
    ap.add_argument("--baseline-only", action="store_true",
                    help="report §25.1's power inputs from existing campaigns")
    args = ap.parse_args()

    import run as locbench
    frame = None
    if args.frame:
        frame = set(json.loads(args.frame.read_text())["instances"])
    data = collect(args.runs,
                   lambda c: locbench.ARM_TOOL.get(c, "rg" if c == "rg" else "semgrep"),
                   frame if not args.baseline_only else None)

    if args.baseline_only:
        print("§25.1 power inputs, recomputed from the transcripts on disk:\n")
        for cond in ("desc-v9", "desc-v5", "rg"):
            xs = [v[cond]["reads_after_search"] for v in data.values() if cond in v]
            if len(xs) < 20:
                continue
            print(f"  {cond:9s} n={len(xs):5d}  reads-after-search "
                  f"mean {st.mean(xs):.2f}  sd {st.pstdev(xs):.2f}")
        p = paired(data, "rg", "desc-v5", "reads_after_search")
        print(f"\n  paired sd(diff), desc-v5 − rg: {st.pstdev([y - x for x, y in p]):.2f} "
              f"over {len(p)} paired runs")
        return

    arms = args.arms.split(",")
    print(f"=== §25 display campaign — base={args.base} ===")
    print(f"frame: {args.frame.name if args.frame else 'ALL RUNS ON DISK (unfiltered)'}"
          f"  ·  {len(data)} instance-runs")
    # One n for every contrast, or arms that happened to finish more instances
    # look better for having done so.
    common = {k for k, v in data.items() if all(a in v for a in arms)}
    print(f"complete in all {len(arms)} arms: {len(common)}\n")
    data = {k: v for k, v in data.items() if k in common}
    present = {c for v in data.values() for c in v}
    for cand in [a for a in arms if a != args.base]:
        if cand not in present:
            print(f"\n{cand}: no runs on disk")
            continue
        print(f"\n{cand} − {args.base}")
        for key in ("reads_after_search", "reads", "searches",
                    "median_search_bytes", "truncated_searches"):
            print(line(key, paired(data, args.base, cand, key)))


if __name__ == "__main__":
    main()
