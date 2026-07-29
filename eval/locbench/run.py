#!/usr/bin/env python3
"""Loc-Bench localization ablation: rg vs semgrep vs both as the agent's
search tool, with full performance provenance per run.

Usage:
  python3 eval/locbench/run.py --limit 2 --conditions rg,semgrep --model haiku --keep-worktrees
  python3 eval/locbench/run.py --limit 50 --conditions rg,semgrep,both --model sonnet
  python3 eval/locbench/run.py --instances astropy__astropy-12907 --conditions both

Protocol (RESEARCH.md §7 eval #1): checkout the repo at base_commit, run a
headless `claude -p` agent restricted to one search-tool condition, ask for
the files/functions the issue requires modifying, score against the real
fix's edit_functions. rg/semgrep are wrapped in PATH shims (shim.py) that
log per-invocation wall time / output sizes without altering behavior.
Results append to eval/data/locbench/results.jsonl (one row per
instance × condition); aggregate with report.py.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import scoring

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"
SEMGREP = Path(os.environ.get("SEMGREP_BIN", HERE.parent.parent / "target/release/semgrep"))
RG = os.environ.get("RG_BIN", "/opt/homebrew/bin/rg")
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")

UNAVAILABLE = " `grep` and `git` are unavailable in this environment."

TOOL_LINES = {
    "rg": (
        "The only code search tool available is `rg` (ripgrep): "
        "`rg [flags] <regex> [path]` for exact/regex content matching. "
        "Read and Glob are also available."
    ),
    # v3 (one tool, two modes): informative tradeoff context instead of a
    # prescriptive routing rule — §7.2 showed agents obey rules mechanically,
    # so a miscalibrated rule is worse than letting the agent decide from
    # context. Includes a micro-example (models imitate examples better than
    # they follow rules). Parameterized by name for the name-gravity ablation.
    "semgrep": None,  # filled below via tool_line()
}


def tool_line(name):
    return (
        f"The only code search tool available is `{name}` — one tool, two "
        f"modes. Ranked mode: `{name} \"plain language query\" [path]` "
        "returns the 10 most relevant locations as path:line:text "
        f"(`-k N` for more). Exact mode: `{name} -e <regex> [path]` is "
        "grep semantics — every literal match, exit 1 on none. Exact mode "
        "shines when you know the precise symbol or string and need every "
        "occurrence, or proof of absence. Ranked mode shines when you know "
        "the behavior but not the vocabulary — it finds code whose words "
        "you can't guess, so rephrasing beats retrying spelling variants. "
        f"Example: {name} \"where is the retry backoff computed\" → "
        "src/net/retry.rs:142:fn backoff_delay(attempt: u32). "
        "Read and Glob are also available."
    )


TOOL_LINES["semgrep"] = tool_line("semgrep")
# Name-gravity ablation: identical binary, identical description — only the
# tool's name changes. If "semgrep" imports grep habits, "search" shouldn't.
TOOL_LINES["search"] = tool_line("search")
# Guided routing (v2, replaces "use whichever tool fits the query"): the
# pilot showed agents pick rg 82% of the time on habit when given no
# criteria — this line tests whether an explicit routing rule fixes that.
TOOL_LINES["both"] = (
    "Available code search tools: `rg` (ripgrep) for exact/regex content "
    "matching, and `semgrep`, a ranked hybrid code search. Routing: for an "
    "exact symbol, string, or regex you already know (from the issue text "
    "or a traceback), use `rg`. For behavior, concept, or description "
    "queries — \"where is X handled\", \"what validates Y\" — use "
    "`semgrep \"plain language query\" [path]` first; it returns the most "
    "relevant locations as path:line:text (ranked top 10; `-k N` for "
    "more). If rg returns nothing or floods, switch to a semgrep "
    "plain-language query rather than retrying spelling variants. "
    "Read and Glob are also available."
)
TOOL_LINES = {k: v + UNAVAILABLE for k, v in TOOL_LINES.items()}

# v4 description (§7.3 winner: ranked-as-identity framing + micro-example).
# Held FIXED across the §9.6 A/B engine conditions so the engine config is
# the only variable; the agent never sees the injected flags.
V4_LINE = (
    "The only code search tool available is `semgrep`, a ranked hybrid "
    "code search. Give it anything — an identifier, a phrase, or a "
    "question: `semgrep \"query\" [path]` returns the k most relevant "
    "locations as path:line:text (ranked top 10; `-k N` for more). "
    "Example: semgrep \"where is the retry backoff computed\" → "
    "src/net/retry.rs:142:fn backoff_delay(attempt: u32). Ranked, not "
    "exhaustive — if the answer isn't there, rephrase. Use `semgrep -e "
    "<regex>` for exact grep-style matching when you need every "
    "occurrence. Read and Glob are also available."
)

# A/B engine conditions: name -> semgrep flags injected by the shim.
# sg-sif additionally gets a --sif --sif-a 1e-4 index (built last per
# instance since it replaces the worktree's .semgrep).
SG_ENGINE_CONDITIONS = {
    "sg-plain": "",
    "sg-mx48": "--maxsim --maxsim-pool 48",
    "sg-mx96": "--maxsim",
    "sg-sif": "--maxsim",
    # Retired candidates (results retained in eval/data/locbench/, analysis
    # via table-ab.py): sg-code (code-distilled table, §10.6) and sg-fnchunk
    # (tree-sitter function chunking, §11) both lost and were removed from
    # the tree. sg-p256 became the shipped default, so it is now sg-plain.
}
for _name in SG_ENGINE_CONDITIONS:
    TOOL_LINES[_name] = V4_LINE + UNAVAILABLE

ALLOWED = {
    "rg": ["Bash(rg *)"],
    "semgrep": ["Bash(semgrep *)"],
    "search": ["Bash(search *)"],
    "both": ["Bash(rg *)", "Bash(semgrep *)"],
    **{name: ["Bash(semgrep *)"] for name in SG_ENGINE_CONDITIONS},
}

PROMPT = """You are localizing where a change must be made in the repository at {tree}
(your current working directory). Do not modify anything and do not use git.

<issue>
{issue}
</issue>

Identify the files and functions that must be modified to resolve this issue.
Explore with your available search tool(s) and Read to verify candidates.

When confident, end your reply with ONLY this JSON (no code fence, no text after it):
{{"files": ["relative/path.py", ...], "functions": ["relative/path.py:ClassName.method_name", "relative/path.py:function_name", ...]}}
Rules: paths relative to the repo root; at most 5 files and 10 functions, ordered most-likely first."""

emit_lock = threading.Lock()
repo_locks = defaultdict(threading.Lock)


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------

def repo_key(repo, sha):
    return f"{repo.replace('/', '__')}__{sha[:12]}"


def ensure_worktree(repo, sha):
    """Blobless bare mirror per repo + detached worktree per (repo, sha).
    Returns (tree_path, meta dict). Meta lives outside the tree so the
    checkout stays pristine."""
    mirrors = DATA / "repos" / "mirrors"
    trees = DATA / "repos" / "trees"
    metas = DATA / "repos" / "meta"
    for d in (mirrors, trees, metas):
        d.mkdir(parents=True, exist_ok=True)
    mirror = mirrors / f"{repo.replace('/', '__')}.git"
    tree = trees / repo_key(repo, sha)
    meta_path = metas / f"{repo_key(repo, sha)}.json"

    with repo_locks[repo]:
        if not mirror.exists():
            subprocess.run(
                ["git", "clone", "--bare", "--filter=blob:none",
                 f"https://github.com/{repo}.git", str(mirror)],
                check=True, capture_output=True, timeout=1800,
            )
        if not tree.exists():
            t0 = time.perf_counter()
            have = subprocess.run(
                ["git", "-C", str(mirror), "cat-file", "-e", f"{sha}^{{commit}}"],
                capture_output=True,
            )
            if have.returncode != 0:
                subprocess.run(["git", "-C", str(mirror), "fetch", "origin", sha],
                               check=True, capture_output=True, timeout=1800)
            subprocess.run(["git", "-C", str(mirror), "worktree", "add", "--detach",
                            str(tree), sha], check=True, capture_output=True, timeout=1800)
            # Quarantine any agent config the target repo ships: it must not
            # steer or contaminate the eval agent.
            had_claude_md = False
            for name in ("CLAUDE.md", ".claude"):
                p = tree / name
                if p.exists():
                    had_claude_md = True
                    p.rename(tree.parent / f"{tree.name}.quarantine.{name.strip('.')}")
            n_files, n_bytes = 0, 0
            for root, dirs, files in os.walk(tree):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    n_files += 1
                    try:
                        n_bytes += (Path(root) / f).stat().st_size
                    except OSError:
                        pass
            meta_path.write_text(json.dumps({
                "checkout_s": round(time.perf_counter() - t0, 1),
                "file_count": n_files,
                "tree_mb": round(n_bytes / 1e6, 1),
                "had_claude_md": had_claude_md,
            }))
    return tree, json.loads(meta_path.read_text())


def ensure_index(tree, meta_path, sif=False):
    """Build .semgrep once per worktree (semgrep conditions); record cost.
    `sif` selects the --sif --sif-a 1e-4 variant — a mismatched existing
    index (normal vs sif) is rebuilt, so sif conditions should run last.
    An index older than the binary is also rebuilt: a rebuilt binary may
    embed different dims (e.g. a swapped embedding table), and reusing that
    index makes every query bail on the dims check — which would look like
    a catastrophic accuracy result rather than the mechanical mismatch it is."""
    meta = json.loads(meta_path.read_text())
    idx_meta = tree / ".semgrep" / "meta.json"
    if idx_meta.exists():
        fresh = idx_meta.stat().st_mtime >= SEMGREP.stat().st_mtime
        if json.loads(idx_meta.read_text()).get("sif", False) == sif and fresh:
            return {"built": False, "reused": True, "sif": sif,
                    "build_wall_s": meta.get("index_build_s"),
                    "index_mb": meta.get("index_mb")}
    t0 = time.perf_counter()
    cmd = [str(SEMGREP), "index", str(tree)]
    if sif:
        cmd += ["--sif", "--sif-a", "0.0001"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    build_s = round(time.perf_counter() - t0, 1)
    index_mb = round(sum(f.stat().st_size for f in (tree / ".semgrep").rglob("*") if f.is_file()) / 1e6, 1)
    meta.update(index_build_s=build_s, index_mb=index_mb)
    meta_path.write_text(json.dumps(meta))
    return {"built": True, "reused": False, "sif": sif,
            "build_wall_s": build_s, "index_mb": index_mb}


def remove_worktree(repo, sha):
    mirror = DATA / "repos" / "mirrors" / f"{repo.replace('/', '__')}.git"
    tree = DATA / "repos" / "trees" / repo_key(repo, sha)
    with repo_locks[repo]:
        subprocess.run(["git", "-C", str(mirror), "worktree", "remove", "--force", str(tree)],
                       capture_output=True)
        subprocess.run(["git", "-C", str(mirror), "worktree", "prune"], capture_output=True)


# ---------------------------------------------------------------------------
# agent invocation
# ---------------------------------------------------------------------------

# Claude Code auto-allows read-only Bash commands, so grep/git can't be kept
# out via --allowedTools. Blocker shims intercept them on PATH: grep-family
# because it would give every condition an un-instrumented search tool, git
# because history after base_commit contains the real fix (`git log --all
# --grep=<issue#>` is straight ground-truth leakage).
BLOCKED_TOOLS = ["grep", "egrep", "fgrep", "git"]


def make_shims(bin_dir):
    """Shim every search-adjacent tool name in every condition; run_agent's
    env decides which get a real binding (the rest hit shim.py's blocked
    path). This also covers the off-condition tool: bare `rg` in the semgrep
    condition must not fall through to the real homebrew binary."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in ["rg", "semgrep", "search", *BLOCKED_TOOLS]:
        w = bin_dir / tool
        w.write_text(f'#!/bin/sh\nexec /usr/bin/env python3 "{HERE / "shim.py"}" {tool} "$@"\n')
        w.chmod(0o755)


def block_msgs(condition):
    steer = {
        "rg": "use `rg` (ripgrep) for content search",
        "semgrep": "use `semgrep \"your query\"` for content search",
        "search": "use `search \"your query\"` for content search",
        "both": "use `rg` or `semgrep` for content search",
        **{n: "use `semgrep \"your query\"` for content search"
           for n in SG_ENGINE_CONDITIONS},
    }[condition]
    msgs = {f"LOCBENCH_BLOCKMSG_{t.upper()}":
            f"{t}: unavailable in this environment — {steer}"
            for t in ("grep", "egrep", "fgrep", "rg", "semgrep", "search")}
    msgs["LOCBENCH_BLOCKMSG_GIT"] = (
        "git: unavailable in this environment — do not retry git commands; "
        "search the working tree instead"
    )
    return msgs


def extract_answer(text):
    """Last balanced top-level {...} containing a "files" key."""
    spans = []
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    for s, e in reversed(spans):
        try:
            obj = json.loads(text[s:e])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "files" in obj:
            files = [str(x) for x in obj.get("files") or []]
            funcs = [str(x) for x in obj.get("functions") or []]
            return files, funcs
    return None


def run_agent(instance, condition, tree, run_dir, args):
    """One claude -p run. Returns (status, answer, agent_stats, search_stats,
    transcript_rel, shim_log_rel)."""
    cond_dir = run_dir / instance["instance_id"] / condition
    bin_dir = cond_dir / "bin"
    searches = cond_dir / "searches"
    shim_log = cond_dir / "shim_log.jsonl"
    transcript = cond_dir / "transcript.jsonl"
    make_shims(bin_dir)

    path = f"{bin_dir}:{os.environ['PATH']}"
    env = os.environ | {
        "PATH": path,
        "LOCBENCH_SHIM_LOG": str(shim_log),
        "LOCBENCH_STDOUT_DIR": str(searches),
        # semgrep's write-through cache: isolate per run dir so eval runs
        # never populate (or read) the user's real ~/.cache/semgrep.
        "SEMGREP_CACHE_DIR": str(run_dir / "semgrep-cache"),
        **block_msgs(condition),
    }
    # Only the condition's tools get a REAL_* binding; the rest of the
    # shimmed names fall through to shim.py's blocked path.
    if condition in ("rg", "both"):
        env["LOCBENCH_REAL_RG"] = RG
    if condition in ("semgrep", "both") or condition in SG_ENGINE_CONDITIONS:
        env["LOCBENCH_REAL_SEMGREP"] = str(SEMGREP)
    if condition == "search":
        env["LOCBENCH_REAL_SEARCH"] = str(SEMGREP)
    flags = SG_ENGINE_CONDITIONS.get(condition, "")
    if flags:
        env["LOCBENCH_SEMGREP_FLAGS"] = flags
    cmd = [
        CLAUDE, "-p", "--output-format", "stream-json", "--verbose",
        "--model", args.model,
        "--tools", "Bash,Read,Glob",
        "--allowedTools", *ALLOWED[condition],
        "--permission-mode", "dontAsk",
        "--setting-sources", "user",
        # Belt and braces vs PATH resets in the Bash tool's login shell.
        "--settings", json.dumps({"env": {"PATH": path}}),
        "--append-system-prompt", TOOL_LINES[condition],
        "--max-budget-usd", str(args.budget_usd),
        "--no-session-persistence",
        PROMPT.format(tree=tree, issue=instance["problem_statement"]),
    ]

    status = "ok"
    t0 = time.perf_counter()
    with open(transcript, "wb") as tf:
        try:
            proc = subprocess.run(cmd, cwd=tree, env=env, stdout=tf,
                                  stderr=subprocess.PIPE, timeout=args.timeout)
            if proc.returncode != 0:
                status = "agent_error"
        except subprocess.TimeoutExpired:
            status = "timeout"
    harness_wall_s = round(time.perf_counter() - t0, 1)

    # --- transcript ---
    agent = {"num_turns": None, "total_cost_usd": None, "duration_ms": None,
             "duration_api_ms": None, "usage": {}, "bash_calls_in_transcript": 0,
             "forbidden_tool_events": 0}
    result_text = ""
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    if block.get("name") == "Bash":
                        agent["bash_calls_in_transcript"] += 1
                    elif block.get("name") in ("Grep", "Task", "Agent"):
                        agent["forbidden_tool_events"] += 1
        elif ev.get("type") == "result":
            agent.update(
                num_turns=ev.get("num_turns"),
                total_cost_usd=ev.get("total_cost_usd"),
                duration_ms=ev.get("duration_ms"),
                duration_api_ms=ev.get("duration_api_ms"),
                usage=ev.get("usage") or {},
            )
            result_text = ev.get("result") or ""
            if ev.get("is_error") and status == "ok":
                status = "agent_error"

    answer = extract_answer(result_text) if result_text else None
    if status == "ok" and answer is None:
        status = "parse_error"

    # --- shim log ---
    search = {"n_invocations": 0, "n_rg": 0, "n_semgrep": 0, "n_search": 0, "n_blocked": 0,
              "blocked_tools": {}, "total_wall_ms": 0.0, "wall_ms_by_tool": {},
              "total_stdout_bytes": 0, "n_nonzero_exit": 0,
              "shim_bypass_suspected": False}
    if shim_log.exists():
        for line in shim_log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("blocked"):
                search["n_blocked"] += 1
                search["blocked_tools"][row["tool"]] = (
                    search["blocked_tools"].get(row["tool"], 0) + 1
                )
                continue
            search["n_invocations"] += 1
            search[f"n_{row['tool']}"] += 1
            search["total_wall_ms"] += row["wall_ms"]
            search["wall_ms_by_tool"][row["tool"]] = (
                search["wall_ms_by_tool"].get(row["tool"], 0.0) + row["wall_ms"]
            )
            search["total_stdout_bytes"] += row["stdout_bytes"]
            search["n_nonzero_exit"] += int(row["exit"] != 0)
    search["total_wall_ms"] = round(search["total_wall_ms"], 1)
    search["wall_ms_by_tool"] = {k: round(v, 1) for k, v in search["wall_ms_by_tool"].items()}
    # Bash ran but the shims never fired (not even a blocked one) -> the
    # agent reached real binaries some other way.
    if (agent["bash_calls_in_transcript"] > 0
            and search["n_invocations"] == 0 and search["n_blocked"] == 0):
        search["shim_bypass_suspected"] = True

    rel = lambda p: str(p.relative_to(DATA))
    return status, answer, agent, search, harness_wall_s, rel(transcript), rel(shim_log)


# ---------------------------------------------------------------------------
# per-instance driver
# ---------------------------------------------------------------------------

def run_instance(instance, conditions, run_dir, args, emit):
    iid = instance["instance_id"]
    repo, sha = instance["repo"], instance["base_commit"]
    try:
        tree, meta = ensure_worktree(repo, sha)
    except subprocess.CalledProcessError as e:
        for condition in conditions:
            emit({"instance_id": iid, "condition": condition, "status": "checkout_error",
                  "repo": repo, "base_commit": sha,
                  "error": (e.stderr or b"").decode(errors="replace")[-500:]})
        return
    meta_path = DATA / "repos" / "meta" / f"{repo_key(repo, sha)}.json"

    gold_files, gold_funcs = scoring.parse_gold(instance["edit_functions"])
    try:
        for condition in conditions:  # rg first by default: it must never see an index
            index = {"built": False, "reused": False, "build_wall_s": None, "index_mb": None}
            if condition in ("semgrep", "both") or condition in SG_ENGINE_CONDITIONS:
                index = ensure_index(tree, meta_path, sif=(condition == "sg-sif"))
            index["present_during_run"] = (tree / ".semgrep").exists()

            status, answer, agent, search, harness_wall_s, transcript_rel, shim_log_rel = run_agent(
                instance, condition, tree, run_dir, args
            )
            answer_files, answer_functions = answer or ([], [])
            metrics = scoring.score_instance(
                answer_files, answer_functions, gold_files, gold_funcs,
                worktree_prefix=str(tree),
            )
            metrics["first_hit_search_seq"] = scoring.first_gold_hit_seq(
                DATA / shim_log_rel, DATA / shim_log_rel.replace("shim_log.jsonl", "searches"),
                gold_files,
            )
            emit({
                "run_id": args.run_id, "instance_id": iid, "condition": condition,
                "model": args.model, "repo": repo, "base_commit": sha,
                "category": instance.get("category"), "status": status,
                "answer_files": answer_files, "answer_functions": answer_functions,
                "gold_files": gold_files,
                "gold_functions": [f"{p}:{q}" for p, q in gold_funcs],
                "metrics": metrics, "agent": agent, "search": search, "index": index,
                "checkout": meta,
                "timing": {"harness_wall_s": harness_wall_s,
                           "ts_end": time.strftime("%Y-%m-%dT%H:%M:%S")},
                "paths": {"transcript": transcript_rel, "shim_log": shim_log_rel},
            })
    finally:
        if not args.keep_worktrees:
            remove_worktree(repo, sha)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def stratified_sample(rows, limit, seed):
    """Deterministic sample spread across categories, then repos within each."""
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("category") or "?"].append(r)
    for rs in by_cat.values():
        rng.shuffle(rs)
        rs.sort(key=lambda r: r["repo"])  # interleave repos after shuffle
    picked, cats = [], sorted(by_cat)
    while len(picked) < limit:
        for c in cats:
            if by_cat[c] and len(picked) < limit:
                picked.append(by_cat[c].pop(0))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--conditions", default="rg,semgrep,both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--instances", default="", help="comma-separated instance_ids (overrides --limit)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip (instance, condition, model) rows already ok in --out")
    ap.add_argument("--out", type=Path, default=DATA / "results.jsonl")
    args = ap.parse_args()

    if not SEMGREP.exists():
        sys.exit(f"build first: cargo build --release   (missing {SEMGREP})")
    if not Path(RG).exists():
        sys.exit(f"missing rg at {RG} (set RG_BIN)")
    if not args.dataset.exists():
        sys.exit(f"missing dataset: run  bash eval/locbench/fetch-dataset.sh")
    subprocess.run([CLAUDE, "--version"], check=True, capture_output=True)

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines()]
    if args.instances:
        want = set(args.instances.split(","))
        rows = [r for r in rows if r["instance_id"] in want]
    else:
        rows = stratified_sample(rows, args.limit, args.seed)
    conditions = args.conditions.split(",")

    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add((r["instance_id"], r["condition"], r.get("model")))

    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = DATA / "runs" / args.run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "a")

    n_done = 0

    def emit(record):
        nonlocal n_done
        with emit_lock:
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_done += 1
            m = record.get("metrics") or {}
            print(f"[{n_done}/{len(rows) * len(conditions)}] {record['instance_id']} "
                  f"{record['condition']}: {record['status']}"
                  f" file_acc@5={m.get('file_acc@5')}"
                  f" searches={((record.get('search') or {}).get('n_invocations'))}"
                  f" cost=${((record.get('agent') or {}).get('total_cost_usd') or 0):.2f}",
                  flush=True)

    todo = []
    for r in rows:
        conds = [c for c in conditions if (r["instance_id"], c, args.model) not in done]
        if conds:
            todo.append((r, conds))
    print(f"{len(todo)} instances × {conditions} (model={args.model}, run={args.run_id})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda t: run_instance(t[0], t[1], run_dir, args, emit), todo))

    out_f.close()
    print(f"wrote {n_done} rows to {args.out}")


if __name__ == "__main__":
    main()
