#!/usr/bin/env python3
"""Pack a campaign's results and trajectories into one reviewable bundle.

Every finding in this project came from reading trajectories — the §16.11
file-scope bug, the `index` false affordance, `built_but_missed` — and every
one of them took a person opening `shim_log.jsonl` by hand. The evidence sits
in four formats inside a gitignored 408 MB tree, so nobody without this
checkout can see any of it.

This flattens the parts a reviewer needs into a single JSON file that
`viewer.py` renders. Extraction rather than wholesale copying is what makes it
fit: transcripts are 11.5 MB raw but only 0.36 MB of actual text.

    python3 eval/locbench/capture.py                      # defaults below
    python3 eval/locbench/capture.py --full results-scale.jsonl

Two levels, because trajectories cost ~26 KB per run and summaries cost ~400 B:

  --full   results files whose runs get complete trajectories
  --summary  results files that contribute outcome rows only, for context

Truncation is capped (see CAPS) and **every cap records what it dropped**. A
viewer that silently hides a search would reproduce the exact failure this
project keeps having.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# `semgrep` or `rg` in *command* position: at the start, or after a pipe,
# semicolon, `&&`, or an env-var prefix. Deliberately not a substring test —
# see the comment at its use site in `timeline_of`.
INVOKES_SEARCH = re.compile(r"(?:^|[;&|]\s*|\b[A-Z_]+=\S+\s+)(semgrep|rg)\s")

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"

# Per-run limits. Generous enough that nothing in tier 1 is actually cut (the
# largest run made 24 searches), tight enough that a 1,115-run campaign would
# still produce a openable page if someone passes one to --full.
CAPS = {
    "searches": 40,
    "turns": 50,
    "out_excerpt": 1200,
    "turn_text": 800,
    "thinking": 500,
    # Problem statements run 1.4 KB median but 55 KB at the tail, and the tail
    # is one issue thread pasted whole. Bounded so a handful of outliers cannot
    # dominate the bundle.
    "problem": 4000,
}


def load_tasks(dataset):
    """instance_id -> the task the agent was given.

    Kept out of the results rows on purpose: `problem_statement` lives on the
    dataset, and a reviewer looking at a trajectory cannot judge whether a
    query was reasonable without seeing what was asked.
    """
    out = {}
    for r in read_jsonl(dataset):
        ps = r.get("problem_statement")
        if ps:
            out[r.get("instance_id")] = clip(ps, CAPS["problem"])
    return out


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_results(path):
    """Rows keyed last-wins: --resume re-runs a failed row and appends."""
    rows = {}
    for r in read_jsonl(path):
        rows[(r.get("instance_id"), r.get("condition"), r.get("model"))] = r
    return list(rows.values())


def clip(text, n):
    """Truncate, and say so in-band so the page cannot imply it has it all."""
    if text is None:
        return None
    if len(text) <= n:
        return text
    return text[:n] + f"\n… [{len(text) - n:,} more chars]"


def cond_dir(row):
    return DATA / "runs" / row["run_id"] / row["instance_id"] / row["condition"]


def run_key(tier, instance_id, condition):
    """The identity of one run inside the bundle.

    Tier belongs in it: samples overlap (tier-1a and tier-1b share 7
    instances), so instance/condition alone is not unique across tiers.
    """
    return f"{tier}::{instance_id}::{condition}"


def summary_row(r):
    """The outcome, without the trajectory. ~400 bytes."""
    m = r.get("metrics") or {}
    a = r.get("agent") or {}
    s = r.get("search") or {}
    return {
        "instance_id": r.get("instance_id"),
        "condition": r.get("condition"),
        "status": r.get("status"),
        "repo": r.get("repo"),
        "category": r.get("category"),
        "cost": round(a.get("total_cost_usd") or 0, 4),
        "turns": a.get("num_turns"),
        "wall_s": (r.get("timing") or {}).get("harness_wall_s"),
        "n_searches": s.get("n_invocations"),
        "n_semgrep": s.get("n_semgrep"),
        "n_rg": s.get("n_rg"),
        "n_blocked": s.get("n_blocked"),
        # Every metric, not a chosen few: which endpoint matters has changed
        # three times in this project and a bundle that pre-selects would have
        # to be regenerated each time.
        "metrics": {k: v for k, v in m.items()},
        "gold_files": r.get("gold_files") or [],
        "gold_functions": r.get("gold_functions") or [],
        "answer_files": r.get("answer_files") or [],
        "answer_functions": r.get("answer_functions") or [],
        "has_trajectory": False,
        # Filled by denials_of. Not from `search`, because the harness cannot
        # see these at all — see that function.
        "n_denied": None,
    }


# Claude Code's own refusal when a Bash call is not covered by --allowedTools.
# Distinct from the shim's block message, which names the tool and steers.
_DENIAL = "Permission to use Bash has been denied"


def denials_of(d):
    """How many tool calls the *permission layer* refused, which no other
    record holds.

    A denied call never executes, so the shim never runs, so `shim_log.jsonl`
    has no row and `search.n_invocations` cannot count it. The loss is real:
    Claude Code evaluates a compound command as a whole, so an agent that types
    `rg …; rg …; git log …` under `Bash(rg *)` has *both* searches refused
    because of the `git`. On the §19.7 campaign that cost 288 calls across 88
    tasks, and 29 tasks per arm ended with zero searches actually run.

    Counted from the transcript because that is the only place it appears.
    """
    t = d / "transcript.jsonl"
    if not t.exists():
        return None
    try:
        return t.read_text(errors="replace").count(_DENIAL)
    except OSError:
        return None


def searches_of(row, d):
    """Each tool invocation: what was typed, what came back, what the engine did.

    Joins three sources the harness writes separately — the shim log (argv,
    exit), `searches/NNN.out` (the bytes the agent actually saw), and the trace
    envelope (mode, files_walked, chunks, hits). Reading them together is what
    turns "returned nothing" into "walked 1 file and could not read it".
    """
    shim = read_jsonl(d / "shim_log.jsonl")
    traces = [t for t in read_jsonl(d / "trace.jsonl") if t.get("kind") == "search"]
    # The trace has no seq, so pair positionally against the non-blocked
    # semgrep invocations, which is the order both are appended in. An
    # exact-miss suggestion emits a second envelope, so consume by query.
    t_by_query = {}
    for t in traces:
        t_by_query.setdefault((t.get("input") or {}).get("query"), []).append(t)

    out, dropped, pos = [], 0, 0
    for e in sorted(shim, key=lambda r: r.get("seq") or 0):
        if e.get("blocked"):
            continue
        # 1-based position among non-blocked invocations, which is what
        # `scoring.first_gold_hit_seq` counts. The shim's own `seq` includes
        # blocked git/grep calls, so comparing against it silently never
        # matches and the "first gold hit" marker just never appears — a
        # wrong answer that looks like an absent one.
        pos += 1
        if len(out) >= CAPS["searches"]:
            dropped += 1
            continue
        argv = e.get("argv") or []
        rec = {
            "seq": e.get("seq"),
            "pos": pos,
            "tool": e.get("tool"),
            "argv": argv,
            "exit": e.get("exit"),
            "stdout_bytes": e.get("stdout_bytes"),
            "wall_ms": e.get("wall_ms"),
        }
        f = e.get("stdout_file")
        if f:
            p = d / "searches" / f
            if p.exists():
                rec["out"] = clip(p.read_text(errors="replace"), CAPS["out_excerpt"])
            err = p.with_suffix(".err")
            if err.exists():
                rec["err"] = clip(err.read_text(errors="replace"), 400)
        # The engine's own account of this search, when there is one.
        q = next((a for a in argv if not a.startswith("-")), None)
        bucket = t_by_query.get(q)
        if bucket:
            t = bucket.pop(0)
            res, inp, res2 = t.get("results") or {}, t.get("input") or {}, t.get("resolution") or {}
            rec["trace"] = {
                "mode": inp.get("mode"),
                "path_taken": res2.get("path_taken"),
                "used_index": res2.get("used_index"),
                "wrote_cache": res2.get("wrote_cache"),
                "files_walked": res.get("files_walked"),
                "n_chunks_considered": res.get("n_chunks_considered"),
                "n_hits": res.get("n_hits"),
                "total_ms": round((t.get("timing") or {}).get("total_ms") or 0, 1),
            }
        out.append(rec)
    return out, dropped


def _result_text(content):
    """A tool_result's payload as text. It arrives as a string or as blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text")
    return "" if content is None else str(content)


def timeline_of(d, searches):
    """What happened, in the order it happened.

    The transcript is the only record that has the ordering *and* the agent's
    reasoning, so the timeline is built from it rather than from the shim log.
    `tool_use.id` and `tool_result.tool_use_id` pair exactly, which is what lets
    a call and its response sit together instead of in two separate lists the
    reader has to match up by hand.

    Every tool is kept, not only searches: the agent also runs `Read` and
    `Glob`, and "searched, then opened *this* file" is most of the story about
    whether a search actually helped.

    Engine facts are joined onto the semgrep calls in invocation order — the
    shim log and the transcript record the same calls in the same sequence — so
    mode / files walked / chunks / hits land under the result they explain.
    """
    entries = read_jsonl(d / "transcript.jsonl")
    # tool_use_id -> result, from the user turns that carry them.
    results = {}
    for e in entries:
        if e.get("type") != "user":
            continue
        for c in (e.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                results[c.get("tool_use_id")] = c

    search_q = list(searches)          # consumed in order by semgrep/rg calls
    out, dropped, unrecorded = [], 0, 0
    for e in entries:
        if e.get("type") != "assistant":
            continue
        for c in (e.get("message") or {}).get("content") or []:
            kind = c.get("type")
            if kind not in ("text", "thinking", "tool_use"):
                continue
            # 73% of thinking blocks carry a signature and no text — Claude Code
            # does not persist the reasoning itself in these transcripts. They
            # are counted, not emitted: 734 blank rows across the bundle would
            # be noise, but "the agent reasoned here and it was not recorded" is
            # worth saying once per run rather than pretending it never thought.
            if kind in ("text", "thinking"):
                body = c.get("text") if kind == "text" else c.get("thinking")
                if not (body or "").strip():
                    unrecorded += 1
                    continue
            if len(out) >= CAPS["turns"]:
                dropped += 1
                continue
            if kind == "text":
                out.append({"kind": "text", "text": clip(c.get("text", ""), CAPS["turn_text"])})
            elif kind == "thinking":
                out.append({"kind": "thinking",
                            "text": clip(c.get("thinking", ""), CAPS["thinking"])})
            else:
                cmd = (c.get("input") or {}).get("command")
                item = {
                    "kind": "tool_use",
                    "name": c.get("name"),
                    "input": clip(cmd or json.dumps(c.get("input") or {}), CAPS["turn_text"]),
                }
                res = results.get(c.get("id"))
                if res is not None:
                    item["result"] = clip(_result_text(res.get("content")), CAPS["out_excerpt"])
                    item["is_error"] = bool(res.get("is_error"))
                # A shell call that *invoked* the search tool gets the engine's
                # own account of it, matched by order.
                #
                # The test has to be "is this program semgrep/rg", not "does the
                # string appear": every worktree path in this campaign contains
                # `/semgrep/`, so a substring check matched `head -100
                # /…/semgrep/eval/…/test.py` and popped a real search off the
                # queue for it. The result was engine facts — mode, files
                # walked, chunks, hits — displayed under a `head` command, and
                # every search after it attributed to the wrong call.
                # …and one shell call can invoke it more than once:
                # `semgrep "a" x; echo ---; semgrep "b" y` is a single tool_use
                # and two logged searches. Popping one per call desynced the
                # queue, so every later call in that run showed the previous
                # search's engine facts — silently, and plausibly.
                n_invocations = len(INVOKES_SEARCH.findall(cmd or ""))
                if n_invocations and search_q:
                    item["searches"] = [
                        {k: s.get(k) for k in
                         ("pos", "exit", "stdout_bytes", "wall_ms", "trace")}
                        for s in (search_q.pop(0) for _ in
                                  range(min(n_invocations, len(search_q))))]
                out.append(item)
    # Searches the shim logged that no timeline entry claimed. The agent can
    # invoke the tool in shell forms this does not recognise (subshells,
    # backticks), and the transcript can end before the shim log does. They stay
    # visible in the Searches view; the count is carried so the timeline cannot
    # imply it showed everything.
    placed = {s["pos"] for e in out for s in (e.get("searches") or [])}
    unplaced = sum(1 for s in searches if s["pos"] not in placed)
    return out, dropped, unrecorded, unplaced


def provenance_of(d):
    p = d / "meta.json"
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: m.get(k) for k in
            ("model", "semgrep_sha256", "claude_version", "dataset_sha256",
             "tool_line_sha256", "tool_line", "budget_usd", "ts")}


def gate_for(results_path):
    """Run the real gate rather than re-deriving its verdict here.

    The page must not be able to disagree with `triage.py`; the way to
    guarantee that is to show what `triage.py` said.
    """
    out = DATA / f".gate-{results_path.stem}.json"
    try:
        subprocess.run(
            [sys.executable, str(HERE / "triage.py"), "--results", str(results_path),
             "--json", str(out)],
            capture_output=True, timeout=600, check=False)
        if out.exists():
            g = json.loads(out.read_text())
            out.unlink()
            return g
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        pass
    return None


def capture(full, summary, tasks):
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caps": CAPS,
        "tiers": [],
        "runs": {},
        "gates": {},
        "provenance": {},
        # Only for instances that have a trajectory: the summary-only tiers are
        # 1,115 rows and would add ~1.5 MB of text nothing renders.
        "tasks": {},
    }
    for path, want_traj in [(p, True) for p in full] + [(p, False) for p in summary]:
        if not path.exists():
            print(f"  skip {path.name}: not found")
            continue
        rows = load_results(path)
        tier = {"name": path.stem, "file": path.name, "n_rows": len(rows),
                "with_trajectories": want_traj, "rows": []}
        n_traj = 0
        for r in rows:
            sr = summary_row(r)
            d = cond_dir(r) if r.get("run_id") else None
            # Counted for summary tiers too: one integer, and it is the only
            # record of a refused call anywhere in the harness.
            if d and d.exists():
                sr["n_denied"] = denials_of(d)
            if want_traj and d and d.exists() and r.get("status") == "ok":
                searches, drop_s = searches_of(r, d)
                timeline, drop_t, unrec, unplaced = timeline_of(d, searches)
                # Keyed by TIER as well as instance and condition. Two samples
                # can share an instance — tier-1a and tier-1b overlap by 7 —
                # and an instance/condition key silently overwrote 14 of
                # tier-1a's trajectories with tier-1b's. Exactly the quiet data
                # loss this bundle exists to make visible, so `run_key` is the
                # one place it is constructed and the collision check below is
                # a hard failure.
                key = run_key(path.stem, r["instance_id"], r["condition"])
                if key in bundle["runs"]:
                    raise SystemExit(f"duplicate run key {key!r} — two rows claim "
                                     f"the same tier/instance/condition")
                sr["run_key"] = key
                bundle["runs"][key] = {
                    "tier": path.stem,
                    "run_id": r["run_id"],
                    # `searches` stays alongside `timeline`: the Searches view
                    # and the exit-code filters read a flat list, and the
                    # timeline is a narrative rather than an index.
                    "searches": searches,
                    "timeline": timeline,
                    "dropped": {"searches": drop_s, "turns": drop_t,
                                "unrecorded_reasoning": unrec,
                                "searches_off_timeline": unplaced},
                    "provenance": provenance_of(d),
                }
                sr["has_trajectory"] = True
                n_traj += 1
                if r["instance_id"] in tasks:
                    bundle["tasks"][r["instance_id"]] = tasks[r["instance_id"]]
                if not bundle["provenance"].get(path.stem):
                    bundle["provenance"][path.stem] = provenance_of(d)
            tier["rows"].append(sr)
        tier["n_trajectories"] = n_traj
        bundle["tiers"].append(tier)
        g = gate_for(path)
        if g:
            bundle["gates"][path.stem] = g
        print(f"  {path.name}: {len(rows)} rows, {n_traj} trajectories"
              f"{', gate ' + ('PASS' if g and g.get('passed') else 'FAIL') if g else ''}")
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", nargs="*", default=["results-tier1.jsonl",
                                                  "results-tier1b.jsonl"],
                    help="results files whose runs get full trajectories")
    ap.add_argument("--summary", nargs="*", default=["results-scale.jsonl"],
                    help="results files contributing outcome rows only")
    ap.add_argument("--out", type=Path, default=DATA / "viewer-bundle.json")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    args = ap.parse_args()

    full = [DATA / f if not Path(f).is_absolute() else Path(f) for f in args.full]
    summ = [DATA / f if not Path(f).is_absolute() else Path(f) for f in args.summary]
    tasks = load_tasks(args.dataset)
    print(f"capturing {len(full)} full + {len(summ)} summary results files "
          f"({len(tasks)} task statements available)")
    bundle = capture(full, summ, tasks)

    # Round-trip check before writing: every row that claimed a trajectory must
    # have one, and vice versa. Cheap, and it is what caught the key collision.
    claimed = sum(1 for t in bundle["tiers"] for r in t["rows"] if r["has_trajectory"])
    if claimed != len(bundle["runs"]):
        raise SystemExit(f"bundle inconsistent: {claimed} rows claim a trajectory "
                         f"but {len(bundle['runs'])} are stored")

    args.out.write_text(json.dumps(bundle, separators=(",", ":")))
    mb = args.out.stat().st_size / 1e6
    total = sum(t["n_rows"] for t in bundle["tiers"])
    traj = len(bundle["runs"])
    print(f"\nwrote {args.out} — {mb:.2f} MB, {total} rows, {traj} trajectories")
    if mb > 8:
        print("  WARNING: over 8 MB; tighten CAPS rather than shipping a page "
              "that takes ten seconds to open.")


if __name__ == "__main__":
    main()
