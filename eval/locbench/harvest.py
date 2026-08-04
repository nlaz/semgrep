#!/usr/bin/env python3
"""Lossless export of every real agent search from the shim logs (§16.3).

replay.py's harvest is deliberately lossy — it drops rg, drops scopes,
dedupes away frequency and order — because its job is scoring ranked
queries. This one's job is the guess corpus: one output row per invocation,
nothing collapsed, every residual explained. The reconciliation gate exists
because a silent parse failure here becomes a missing stratum that reads as
"agents never type that."

    python3 eval/locbench/harvest.py                    # all runs
    python3 eval/locbench/harvest.py --runs runs/<id>   # one capture run

Output rows (strings and repo-relative paths only — no repo content):
    {run_id, instance_id, condition, tool, mode, kind, seq, ts, pattern,
     patterns, flags, k, scopes_rel, scoped, exit, stdout_bytes, wall_ms}
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

DATA = HERE.parent / "data" / "locbench"

SEARCH_TOOLS = ("semgrep", "search", "sg", "rg")

# semgrep CLI flags that take a value (superset of replay.py's list).
SG_VALUED = {"-k", "-C", "--mode", "--sem-weight", "--mmr-lambda", "--window",
             "--overlap", "--prf", "--maxsim-pool", "--maxsim-blend", "-A", "-B",
             "--embed-preproc", "--sif-a", "--repair-max-drift"}
# rg flags that take a value.
RG_VALUED = {"-g", "--glob", "-t", "--type", "-T", "--type-not", "-A", "-B", "-C",
             "-m", "--max-count", "--color", "-j", "--threads", "-r", "--replace",
             "--max-depth", "--max-filesize", "--sort", "-E", "--encoding"}


def parse_semgrep_argv(argv):
    is_exact, k = False, None
    query, scopes, flags = None, [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-e", "--exact"):
            is_exact = True
        elif a == "-k" and i + 1 < len(argv):
            k = argv[i + 1]
            i += 1
        elif a in SG_VALUED:
            flags.append(f"{a} {argv[i + 1] if i + 1 < len(argv) else ''}".strip())
            i += 1
        elif a.startswith("-"):
            flags.append(a)
        elif query is None:
            query = a
        else:
            scopes.append(a)
        i += 1
    return {"mode": "exact" if is_exact else "ranked",
            "patterns": [query] if query else [], "flags": flags, "k": k,
            "scopes": scopes}


def parse_rg_argv(argv):
    patterns, scopes, flags = [], [], []
    bare = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-e", "--regexp") and i + 1 < len(argv):
            patterns.append(argv[i + 1])
            i += 1
        elif a in RG_VALUED:
            flags.append(f"{a} {argv[i + 1] if i + 1 < len(argv) else ''}".strip())
            i += 1
        elif a.startswith("--") and "=" in a:
            flags.append(a)
        elif a.startswith("-") and a != "-":
            flags.append(a)
        else:
            bare.append(a)
        i += 1
    if not patterns and bare:
        patterns = [bare[0]]
        scopes = bare[1:]
    else:
        scopes = bare
    return {"mode": "rg", "patterns": patterns, "flags": flags, "k": None,
            "scopes": scopes}


def relativize(scope, cwd):
    """Repo-relative scope. The shim records cwd, which is inside the
    worktree (repos/trees/<key>/...). Anything that can't be resolved under
    the tree is returned as-is with scoped-outside semantics."""
    p = PurePosixPath(scope)
    if not p.is_absolute():
        base = PurePosixPath(cwd)
        parts = base.parts
        if "trees" in parts:
            tree_root = PurePosixPath(*parts[: parts.index("trees") + 2])
            sub = base.relative_to(tree_root)
            joined = (sub / p) if str(sub) != "." else p
            # normalize any ./ and ../ best-effort
            norm = []
            for seg in joined.parts:
                if seg == ".":
                    continue
                if seg == ".." and norm:
                    norm.pop()
                else:
                    norm.append(seg)
            return "/".join(norm) or "."
        return str(p)
    parts = p.parts
    if "trees" in parts:
        idx = parts.index("trees")
        rel = parts[idx + 2:]
        return "/".join(rel) or "."
    return str(p)


def harvest(runs_dir):
    rows = []
    residuals = Counter()
    totals = Counter()
    for log in sorted(Path(runs_dir).rglob("shim_log.jsonl")):
        condition = log.parent.name
        instance = log.parent.parent.name
        run_id = log.parent.parent.parent.name
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                residuals["unparseable-log-line"] += 1
                continue
            tool = e.get("tool")
            if tool not in SEARCH_TOOLS:
                continue
            if e.get("blocked"):
                residuals[f"blocked-{tool}"] += 1
                continue
            totals[tool] += 1
            # The shim logs the tool's arguments WITHOUT the tool name
            # (shim.py receives the tool as its own argv[1] and logs "$@").
            argv = e.get("argv") or []
            parsed = parse_rg_argv(argv) if tool == "rg" else parse_semgrep_argv(argv)
            if not parsed["patterns"]:
                residuals[f"empty-pattern-{tool}"] += 1
                continue
            if tool == "rg":
                kind = "guess_rg"
            elif parsed["mode"] == "exact":
                kind = "guess_exact"
            else:
                kind = "guess_ranked"
            cwd = e.get("cwd", "")
            scopes_rel = [relativize(s, cwd) for s in parsed["scopes"]]
            rows.append({
                "run_id": run_id, "instance_id": instance, "condition": condition,
                "tool": tool, "mode": parsed["mode"], "kind": kind,
                "seq": e.get("seq"), "ts": e.get("ts"),
                "pattern": parsed["patterns"][0],
                "patterns": parsed["patterns"],
                "flags": parsed["flags"], "k": parsed["k"],
                "scopes_rel": scopes_rel, "scoped": bool(scopes_rel),
                "exit": e.get("exit"), "stdout_bytes": e.get("stdout_bytes"),
                "wall_ms": e.get("wall_ms"),
            })
    return rows, totals, residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "queries" / "guesses-v0.jsonl")
    args = ap.parse_args()

    rows, totals, residuals = harvest(args.runs)
    rows.sort(key=lambda r: (r["run_id"], r["instance_id"], r["condition"],
                             r["seq"] if r["seq"] is not None else -1))

    by_kind = Counter(r["kind"] for r in rows)
    print("real search invocations by tool:",
          {t: totals[t] for t in SEARCH_TOOLS})
    print("exported rows by kind:", dict(sorted(by_kind.items())))
    explained = {k: v for k, v in residuals.items()
                 if not k.startswith("blocked-")}
    print("explained residuals:", explained or "none")
    # The gate: every real invocation is a row or an explained residual.
    for t in SEARCH_TOOLS:
        exported = sum(1 for r in rows if r["tool"] == t)
        residual = sum(v for k, v in residuals.items()
                       if k == f"empty-pattern-{t}")
        if exported + residual != totals[t]:
            raise SystemExit(
                f"reconciliation failed for {t}: {totals[t]} real calls, "
                f"{exported} exported + {residual} explained = "
                f"{exported + residual}")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
