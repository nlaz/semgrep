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
import hashlib
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

CLAUDE_VERSION = None

UNAVAILABLE = " `grep` and `git` are unavailable in this environment."

# Claude Code's own refusal when a Bash call is not covered by --allowedTools.
# Distinct from the shim's block message, which names the tool and steers. Lives
# here, and `capture.py` imports it, because a marker string defined twice is a
# marker string that eventually differs in one place (RESEARCH.md §19.8).
DENIAL_MARKER = "Permission to use Bash has been denied"

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

# Guess-capture conditions (§16, Phase C): mechanics-only descriptions that
# bound the §7.3 framing effect from two sides instead of pretending a
# neutral description exists. No advice, no tradeoff language, no example —
# §7.3 measured that agents imitate examples. cap-ranked omits `-e` entirely
# (every guess is forced through ranked syntax); cap-two states both modes
# symmetrically. Run with --no-score: capture runs must not mint
# accuracy-shaped numbers under novel descriptions.
CAPTURE_CONDITIONS = {
    "cap-ranked": (
        "The only code search tool available is `semgrep`. Usage: "
        "`semgrep <query> [path]`. It prints up to 10 matching locations as "
        "path:line:text, best matches first (`-k N` for more). "
        "Read and Glob are also available."
    ),
    "cap-two": (
        "The only code search tool available is `semgrep`. Usage: "
        "`semgrep <query> [path]` prints up to 10 locations as "
        "path:line:text, best matches first (`-k N` for more). "
        "`semgrep -e <regex> [path]` prints every line matching the regex, "
        "grep semantics, exit 1 on no match. Read and Glob are also available."
    ),
}
for _name, _line in CAPTURE_CONDITIONS.items():
    TOOL_LINES[_name] = _line + UNAVAILABLE

# Description A/B conditions (§16.7): same engine, same binary, only the
# tool description varies. desc-v4 is the §7.3 winner verbatim; desc-v5
# removes the -e mention entirely (§16.6: mere mention collapses ranked
# usage 72% -> 7%); desc-v6 adds the guess-folding instruction that
# operationalizes §16.5's P6 (ladders belong in one ranked query).
DESC_CONDITIONS = {
    "desc-v4": V4_LINE,
    "desc-v5": (
        "The only code search tool available is `semgrep`, a ranked code "
        "search. Give it anything — an identifier, a phrase, or a question: "
        "`semgrep \"query\" [path]` returns the most relevant locations as "
        "path:line:text (top 10; `-k N` for more). Ranked, not exhaustive — "
        "if the answer isn't there, rephrase. Read and Glob are also "
        "available."
    ),
    "desc-v6": (
        "The only code search tool available is `semgrep`, a ranked code "
        "search: `semgrep \"query\" [path]` returns the most relevant "
        "locations as path:line:text (top 10; `-k N` for more). You don't "
        "need to know what anything is called — describe the behavior, or "
        "if you're torn between several possible names, put ALL your "
        "candidate names in one query and let ranking sort it out. Ranked, "
        "not exhaustive — if the answer isn't there, rephrase. Read and "
        "Glob are also available."
    ),
    # desc-v7 (§19): desc-v5 with the §7.3 micro-example restored, and nothing
    # else — one inserted sentence, character-identical either side of it, so
    # the arm isolates the example rather than a rewrite.
    #
    # The example is in v4 and absent from v5 by accident rather than by
    # decision: v5 was derived by cutting `-e` out of v4 (§16.6), and the
    # example went with it because it was the clause that mentioned a mode.
    # §7.3's winner was ranked-as-identity framing *plus* a micro-example, so
    # what ships today is half of a measured result. §7.3 also found agents
    # imitate examples more reliably than they follow rules, which is what
    # makes an example the lever rather than another sentence of advice.
    "desc-v7": (
        "The only code search tool available is `semgrep`, a ranked code "
        "search. Give it anything — an identifier, a phrase, or a question: "
        "`semgrep \"query\" [path]` returns the most relevant locations as "
        "path:line:text (top 10; `-k N` for more). Example: semgrep \"where "
        "is the retry backoff computed\" → src/net/retry.rs:142:fn "
        "backoff_delay(attempt: u32). Ranked, not exhaustive — if the answer "
        "isn't there, rephrase. Read and Glob are also available."
    ),
    # desc-v8 (§19.2b): desc-v7 with the example's *query* swapped for candidate
    # identifiers, and nothing else — same framing, same worked result line, so
    # v7-vs-v8 isolates the style the example demonstrates rather than whether
    # there is an example at all.
    #
    # §19.2b measured why this is the arm to run: on the agent's own ranked
    # queries, a paraphrase that shares no vocabulary with the gold finds it
    # 4% of the time, against 46% for an identifier guess that also shares
    # none. A static bag-of-words model cannot cross from a description to
    # differently-named code — SIF weights common English words toward zero
    # (a/(a+p)), so a paraphrase reduces to its rare tokens and has nothing
    # left when those miss. desc-v7's example demonstrates that failure mode;
    # this one demonstrates several candidate spellings in one query, which is
    # what raises the chance of the vocabulary overlap that decides the outcome.
    "desc-v8": (
        "The only code search tool available is `semgrep`, a ranked code "
        "search. Give it anything — an identifier, a phrase, or a question: "
        "`semgrep \"query\" [path]` returns the most relevant locations as "
        "path:line:text (top 10; `-k N` for more). Example: semgrep "
        "\"retry_backoff backoff_delay compute_delay\" → "
        "src/net/retry.rs:142:fn backoff_delay(attempt: u32). Ranked, not "
        "exhaustive — if the answer isn't there, rephrase. Read and Glob are "
        "also available."
    ),
    # desc-v9 (§19.9): desc-v8 renamed to `sg`, plus "you run with Bash" folded
    # into the identity sentence rather than added as a standalone line — the
    # failure it targets is an agent emitting a structured tool_use block whose
    # input schema is this description's own signature, so the correction
    # belongs where the tool is introduced, not appended after it.
    #
    # This changes TWO things at once and therefore attributes neither. §16.6
    # and the `search` name-gravity arm both say a name alone can move
    # behaviour, so a later campaign moving is evidence that "v9 moved", not
    # that the Bash clause worked. Accepted deliberately: the defect it
    # addresses is worth 4 tasks in 204, and a frame to split it is not.
    "desc-v9": (
        "The only code search tool available is `sg`, a ranked code search "
        "you run with Bash. Give it anything — an identifier, a phrase, or a "
        "question: `sg \"query\" [path]` returns the most relevant locations "
        "as path:line:text (top 10; `-k N` for more). Example: sg "
        "\"retry_backoff backoff_delay compute_delay\" → "
        "src/net/retry.rs:142:fn backoff_delay(attempt: u32). Ranked, not "
        "exhaustive — if the answer isn't there, rephrase. Read and Glob are "
        "also available."
    ),
    # desc-v10 (§28.2): wide search modeled as the default, the path as a
    # deliberate second step. Every prior variant showed `[path]` in the
    # signature and none said anything about when to use it; agents scoped
    # ~70% of sg calls to guessed paths, and "gold scoped away" cost 17.6% of
    # sg's §28 losses (37.8% of rg's — the habit hurts every tool, and
    # repo-wide ranked search is the surface where sg wins it back). Also
    # corrects the count to the shipped k=5 (desc-v9 still said 10). The
    # example, the mechanism sentence, and the rephrase caveat are desc-v9's
    # own words — the §19 tripwires hold across the change.
    "desc-v10": (
        "The only code search tool available is `sg`, a ranked code search "
        "you run with Bash. Give it anything — an identifier, a phrase, or a "
        "question: `sg \"query\"` searches the whole repository and returns "
        "the most relevant locations as path:line:text (top 5; `-k N` for "
        "more). Start wide: add a path argument only to narrow further after "
        "a wide search has pointed somewhere. Example: sg "
        "\"retry_backoff backoff_delay compute_delay\" → "
        "src/net/retry.rs:142:fn backoff_delay(attempt: u32). Ranked, not "
        "exhaustive — if the answer isn't there, rephrase. Read and Glob are "
        "also available."
    ),
    # desc-v11 (§30.3): v10 plus the exact-match escape hatch, ROUTED. The §30
    # pilot measured sg-arm agents reaching for shell grep 1.2×/session — the
    # single largest cost in the campaign — and the blocked argvs split into
    # exactly two intents: verify/enumerate a known identifier (single-token
    # literals, -e's actual job) and OR-of-candidates regexes
    # (`getCachedChildren\|getCachedTrashed\|…`), which the ranked multi-word
    # query they were already taught serves better. So the clause routes three
    # ways and states the default explicitly.
    #
    # This deliberately walks back §16.10's blanket rule ("never name -e"),
    # with eyes open: that lesson was measured on an UNCONDITIONAL footer
    # mention with the caller's own query interpolated, in an additive regime.
    # Here the mention is conditional ("only to verify or count"), framed
    # inside a ranked-first identity, and the campaign registers exact-share
    # as a tripwire diagnostic — if agents collapse onto -e, that is the §16.10
    # effect reasserting itself and the variant dies as v6 and v7 did.
    "desc-v11": (
        "The only code search tool available is `sg`, a ranked code search "
        "you run with Bash. Give it anything — an identifier, a phrase, or a "
        "question: `sg \"query\"` searches the whole repository and returns "
        "the most relevant locations as path:line:text (top 5; `-k N` for "
        "more). Start wide: add a path argument only to narrow further after "
        "a wide search has pointed somewhere. When you are unsure of a name, "
        "list several candidate spellings in one query rather than a regex. "
        "When you already know the exact string — a name you have seen in "
        "the code, an error message — `sg -e \"the_exact_string\"` returns "
        "every literal match, grep-style. Default to ranked search; use -e "
        "only to verify or count something you have already seen spelled "
        "out. Example: sg \"retry_backoff backoff_delay compute_delay\" → "
        "src/net/retry.rs:142:fn backoff_delay(attempt: u32). Ranked, not "
        "exhaustive — if the answer isn't there, rephrase. Read and Glob are "
        "also available."
    ),
}

# Arms whose agent is told to type something other than `semgrep`. The binary is
# the same either way (`sg` and `semgrep` are two [[bin]] targets over one
# source); only the name in the description and on PATH differs.
# §25's display-format arms. Named here, before ARM_TOOL, because every one of
# them types `sg` and ALLOWED is derived from ARM_TOOL — an arm missing from it
# would have every search denied.
DISPLAY_CONDITIONS = ("disp-line", "disp-full", "disp-head",
                      "pl-1", "pl-18", "pl-full", "pl-18k5")

ARM_TOOL = {"desc-v9": "sg", **{n: "sg" for n in DISPLAY_CONDITIONS}}

# Every search-tool name the shims cover. `sg` is here so its shim exists for
# every arm, not only desc-v9 — an arm that mentions a name with no shim behind
# it would reach the real binary on PATH and escape the harness entirely.
SHIMMED_SEARCH_TOOLS = ("rg", "semgrep", "search", "sg")

for _name, _line in DESC_CONDITIONS.items():
    TOOL_LINES[_name] = _line + UNAVAILABLE

# The display arms inherit desc-v9's line verbatim. Byte-identical across the
# three by construction rather than by copy-paste, which is what §25.1's
# tripwire 7 asserts on the recorded runs.
for _name in DISPLAY_CONDITIONS:
    TOOL_LINES[_name] = TOOL_LINES["desc-v9"]

# A/B engine conditions: name -> semgrep flags injected by the shim.
# sg-sif additionally gets a --sif --sif-a 1e-4 index (built last per
# instance since it replaces the worktree's .semgrep).
SG_ENGINE_CONDITIONS = {
    "sg-plain": "",
    # §25's display arms. The engine flag is injected by the shim and never
    # shown to the agent, so these three differ from each other in exactly one
    # thing: what came back. `disp-line` is the control and is NOT the same as
    # reusing old desc-v9 rows — those came from a pre-§24 binary, and comparing
    # across binaries is the §23.3 contamination trap.
    "disp-line": "",
    "disp-full": "--full",
    "disp-head": "--headers",
    # §26's passage-width arms. Every one names its width explicitly so no arm
    # inherits the shipped default — a control that relied on it would silently
    # become the treatment the day the default moved again.
    "pl-1": "--passage-lines 1",
    "pl-18": "--passage-lines 18",
    "pl-full": "--full",
    "pl-18k5": "--passage-lines 18 -k 5",
    "sg-mx48": "--maxsim --maxsim-pool 48",
    "sg-mx96": "--maxsim",
    "sg-sif": "--maxsim",
    # Retired candidates (results retained in eval/data/locbench/, analysis
    # via table-ab.py): sg-code (code-distilled table, §10.6) and sg-fnchunk
    # (tree-sitter function chunking, §11) both lost and were removed from
    # the tree. sg-p256 became the shipped default, so it is now sg-plain.
}
# §25's display arms carry the SHIPPED description, not V4_LINE: the campaign
# asks what the display format does to the product as it exists, and desc-v9 is
# the product. Their TOOL_LINES are assigned below, once DESC_CONDITIONS exists,
# so there is exactly one copy of that text; this loop covers the older arms.
for _name in SG_ENGINE_CONDITIONS:
    if _name not in DISPLAY_CONDITIONS:
        TOOL_LINES[_name] = V4_LINE + UNAVAILABLE

ALLOWED = {
    "rg": ["Bash(rg *)"],
    "semgrep": ["Bash(semgrep *)"],
    "search": ["Bash(search *)"],
    "both": ["Bash(rg *)", "Bash(semgrep *)"],
    **{name: [f"Bash({ARM_TOOL.get(name, 'semgrep')} *)"] for name in SG_ENGINE_CONDITIONS},
    **{name: ["Bash(semgrep *)"] for name in CAPTURE_CONDITIONS},
    # Derived from tool_of so an arm that types `sg` is permitted `sg`, not
    # `semgrep` — a mismatch here denies every search in the arm.
    **{name: [f"Bash({ARM_TOOL.get(name, 'semgrep')} *)"] for name in DESC_CONDITIONS},
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

def repo_key(repo, sha, iid=None):
    """Worktree identity. `iid` is load-bearing: the dataset contains 28
    instance pairs sharing a (repo, base_commit), and keying the tree by
    (repo, sha) alone let two workers check out, index, and then
    `worktree remove --force` the SAME directory concurrently — deleting a
    tree under a live agent, interleaving two `semgrep index` builds into
    one staging dir, and leaking an index into the rg arm. Found by the
    §16.9 adversarial review before the run, not after."""
    base = f"{repo.replace('/', '__')}__{sha[:12]}"
    return f"{base}__{iid}" if iid else base


def ensure_worktree(repo, sha, iid=None):
    """Blobless bare mirror per repo + detached worktree per (repo, sha).
    Returns (tree_path, meta dict). Meta lives outside the tree so the
    checkout stays pristine."""
    mirrors = DATA / "repos" / "mirrors"
    trees = DATA / "repos" / "trees"
    metas = DATA / "repos" / "meta"
    for d in (mirrors, trees, metas):
        d.mkdir(parents=True, exist_ok=True)
    mirror = mirrors / f"{repo.replace('/', '__')}.git"
    tree = trees / repo_key(repo, sha, iid)
    meta_path = metas / f"{repo_key(repo, sha, iid)}.json"

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
    if not meta_path.exists():
        # An interrupted checkout (Ctrl-C mid-walk) leaves a tree with no
        # meta; every later --resume used to die on FileNotFoundError.
        meta_path.write_text(json.dumps(
            {"checkout_s": None, "file_count": None, "tree_mb": None,
             "had_claude_md": None, "note": "meta reconstructed after an "
                                            "interrupted checkout"}))
    return tree, json.loads(meta_path.read_text())


def ensure_index(tree, meta_path, sif=False, embed_preproc="none"):
    """Build .semgrep once per worktree (semgrep conditions); record cost.
    `sif` selects the --sif --sif-a 1e-4 variant and `embed_preproc` the
    prose-rendering variant — an existing index whose build config differs
    on EITHER is rebuilt, so variant conditions should run last.

    Both must be checked, not just `sif`. The warm path answers a query in
    whatever space the index was built in and ignores the search-side flag
    (search/indexed.rs), so reusing a `none` index for a `split` condition
    does not fail — it silently measures the wrong thing and reports parity.

    An index older than the binary is also rebuilt: a rebuilt binary may
    embed different dims (e.g. a swapped embedding table), and reusing that
    index makes every query bail on the dims check — which would look like
    a catastrophic accuracy result rather than the mechanical mismatch it is."""
    meta = json.loads(meta_path.read_text())
    idx_meta = tree / ".semgrep" / "meta.json"
    want = {"sif": sif, "embed_preproc": embed_preproc}
    if idx_meta.exists():
        have = json.loads(idx_meta.read_text())
        fresh = idx_meta.stat().st_mtime >= SEMGREP.stat().st_mtime
        got = {"sif": have.get("sif", False),
               "embed_preproc": have.get("embed_preproc", "none")}
        if got == want and fresh:
            return {"built": False, "reused": True, **want,
                    "build_wall_s": meta.get("index_build_s"),
                    "index_mb": meta.get("index_mb")}
    t0 = time.perf_counter()
    cmd = [str(SEMGREP), "index", str(tree)]
    if sif:
        cmd += ["--sif", "--sif-a", "0.0001"]
    if embed_preproc != "none":
        cmd += ["--embed-preproc", embed_preproc]
    subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    build_s = round(time.perf_counter() - t0, 1)
    index_mb = round(sum(f.stat().st_size for f in (tree / ".semgrep").rglob("*") if f.is_file()) / 1e6, 1)
    meta.update(index_build_s=build_s, index_mb=index_mb)
    meta_path.write_text(json.dumps(meta))
    return {"built": True, "reused": False, **want,
            "build_wall_s": build_s, "index_mb": index_mb}


def remove_worktree(repo, sha, iid=None):
    """Remove a worktree, and make sure it is gone even if git cannot do it.

    `git worktree remove` needs a live mirror to operate on, and fails quietly
    (capture_output, no check) when there isn't one — so an interrupted run, or
    an eviction that beat the cleanup, leaves the tree on disk with nothing that
    will ever collect it. Four such trees, 218 MB, were found orphaned against
    an empty mirrors directory. On a 560-instance campaign that leak is
    unbounded and would halt the run halfway on a full disk, which is the one
    failure mode that costs a whole chunk of spend rather than one row.

    So: try git first (it also updates the mirror's worktree metadata), then
    verify, then remove the directory outright. Idempotent by construction.
    """
    mirror = DATA / "repos" / "mirrors" / f"{repo.replace('/', '__')}.git"
    tree = DATA / "repos" / "trees" / repo_key(repo, sha, iid)
    with repo_locks[repo]:
        if mirror.exists():
            subprocess.run(["git", "-C", str(mirror), "worktree", "remove", "--force", str(tree)],
                           capture_output=True)
            subprocess.run(["git", "-C", str(mirror), "worktree", "prune"], capture_output=True)
        if tree.exists():
            subprocess.run(["rm", "-rf", str(tree)])


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
    for tool in [*SHIMMED_SEARCH_TOOLS, *BLOCKED_TOOLS]:
        w = bin_dir / tool
        w.write_text(f'#!/bin/sh\nexec /usr/bin/env python3 "{HERE / "shim.py"}" {tool} "$@"\n')
        w.chmod(0o755)


def tool_of(condition):
    """The shimmed binary name this arm's agent is told to run.

    One function rather than a mapping repeated at each site, because the sites
    do not fail the same way. `shim.py` derives `LOCBENCH_REAL_{tool.upper()}`
    from argv, so an arm whose shim name and whose REAL_* binding disagree falls
    into the *blocked* path: exit 2 with a steer message on both streams, which
    looks like a catastrophic result rather than a harness bug. Adding an arm
    that types a new name means adding it here and nowhere else.
    """
    if condition == "rg":
        return "rg"
    if condition == "search":
        return "search"
    return ARM_TOOL.get(condition, "semgrep")


def block_msgs(condition):
    tool = tool_of(condition)
    steer = {
        "rg": "use `rg` (ripgrep) for content search",
        "both": "use `rg` or `semgrep` for content search",
    }.get(condition, f"use `{tool} \"your query\"` for content search")
    # Every shimmed name needs a message, blocked tools included: without one
    # `shim.py` falls back to a bare "X is unavailable" with no steer, which is
    # the silence df9a3c0 went out of its way to end. `git` is overwritten just
    # below with its own wording.
    msgs = {f"LOCBENCH_BLOCKMSG_{t.upper()}":
            f"{t}: unavailable in this environment — {steer}"
            for t in (*BLOCKED_TOOLS, *SHIMMED_SEARCH_TOOLS)}
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
        # No self-teaching footers in an A/B: they advertise `-e` after
        # every ranked search, which would be an uncontrolled co-treatment
        # in exactly the arm whose description withholds it (§16.9 A1).
        "SEMGREP_NO_HINTS": "1",
        # Per-invocation forensics. The §16.10 campaign captured only
        # (argv, exit, stdout_bytes) per search, which is why a bug that
        # silently emptied 47% of the treatment arm was found by reading
        # transcripts weeks later instead of by the harness. This env var is
        # built for exactly this (telemetry.rs): it does not perturb the argv
        # the agent sees, so it works underneath shim.py, and it captures
        # invocations no outer flag can reach — including the second full
        # search an exact-miss suggestion runs. One line per invocation, with
        # mode, index/cache resolution, files_walked, chunks considered,
        # repair outcome, stage timings and exit code. Read by triage.py.
        "SEMGREP_TRACE_FILE": str(cond_dir / "trace.jsonl"),
        **block_msgs(condition),
    }
    # Inherited LOCBENCH_* from the parent shell would silently hand an arm a
    # tool it is supposed to lack; clear before binding (§16.9a hygiene).
    for _t in SHIMMED_SEARCH_TOOLS:
        env.pop(f"LOCBENCH_REAL_{_t.upper()}", None)
        env.pop(f"LOCBENCH_{_t.upper()}_FLAGS", None)
    # Only the condition's tools get a REAL_* binding; the rest of the
    # shimmed names fall through to shim.py's blocked path. Derived from
    # `tool_of` so the name the agent is told to type, the shim it lands on,
    # and the binding that makes it real cannot drift apart.
    if condition in ("rg", "both"):
        env["LOCBENCH_REAL_RG"] = RG
    tool = tool_of(condition)
    if tool != "rg":
        env[f"LOCBENCH_REAL_{tool.upper()}"] = str(SEMGREP)
    flags = SG_ENGINE_CONDITIONS.get(condition, "")
    if flags:
        env[f"LOCBENCH_{tool.upper()}_FLAGS"] = flags
    # Provenance the run dirs never had (§16 C2): the exact appended system
    # prompt was invisible after the fact, so description versions were only
    # recoverable by git archaeology. Written BEFORE the agent starts.
    tool_line_text = TOOL_LINES[condition]
    # Provenance for a multi-day campaign (§16.9a #11): a `cargo build` or a
    # settings edit between chunks would otherwise change the treatment with
    # nothing in the data recording it.
    def _sha(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            return None
    (cond_dir / "meta.json").write_text(json.dumps({
        "instance_id": instance["instance_id"], "condition": condition,
        "model": args.model, "tool_line": tool_line_text,
        "tool_line_sha256": hashlib.sha256(tool_line_text.encode()).hexdigest()[:16],
        "allowed_tools": ALLOWED[condition], "budget_usd": args.budget_usd,
        "semgrep_sha256": _sha(SEMGREP),
        "semgrep_mtime": SEMGREP.stat().st_mtime if SEMGREP.exists() else None,
        "claude_version": CLAUDE_VERSION,
        "settings_sha256": _sha(Path.home() / ".claude" / "settings.json"),
        "dataset_sha256": _sha(args.dataset),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1) + "\n")
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
    n_denied = 0
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
        # Refusals by the permission layer, which no other record can hold: the
        # call never executes, so the shim never runs and n_invocations cannot
        # see it. Counted here rather than in a second pass because this loop
        # already visits every event (RESEARCH.md §19.8).
        if DENIAL_MARKER in line:
            n_denied += line.count(DENIAL_MARKER)
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
    # Per-tool counters derived from the shim list: `search[f"n_{tool}"] += 1`
    # below would KeyError on a tool with no slot, failing the run outright.
    search = {**{f"n_{t}": 0 for t in SHIMMED_SEARCH_TOOLS},
              "n_invocations": 0, "n_blocked": 0,
              "blocked_tools": {}, "total_wall_ms": 0.0, "wall_ms_by_tool": {},
              "total_stdout_bytes": 0, "n_nonzero_exit": 0,
              # Not from the shim log — see the transcript loop above.
              "n_denied": n_denied,
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
        tree, meta = ensure_worktree(repo, sha, iid)
    except subprocess.CalledProcessError as e:
        for condition in conditions:
            emit({"instance_id": iid, "condition": condition,
                  "status": "checkout_error", "model": args.model,
                  "run_id": args.run_id, "repo": repo, "base_commit": sha,
                  "error": (e.stderr or b"").decode(errors="replace")[-500:]})
        return
    meta_path = DATA / "repos" / "meta" / f"{repo_key(repo, sha, iid)}.json"

    gold_files, gold_funcs = scoring.parse_gold(instance["edit_functions"])
    try:
        for condition in conditions:  # rg first by default: it must never see an index
            if args.stop_event is not None and args.stop_event.is_set():
                break
            index = {"built": False, "reused": False, "build_wall_s": None, "index_mb": None}
            if (condition in ("semgrep", "both") or condition in SG_ENGINE_CONDITIONS
                    or condition in CAPTURE_CONDITIONS
                    or condition in DESC_CONDITIONS):
                try:
                    index = ensure_index(tree, meta_path, sif=(condition == "sg-sif"))
                except subprocess.SubprocessError as e:
                    # An index failure is one condition's problem, not the
                    # invocation's: it used to escape run_instance, drop every
                    # remaining row silently, and re-raise out of pool.map.
                    emit({"instance_id": iid, "condition": condition,
                          "status": "index_error", "model": args.model,
                          "run_id": args.run_id, "repo": repo, "base_commit": sha,
                          "error": str(e)[-500:]})
                    continue
            index["present_during_run"] = (tree / ".semgrep").exists()

            status, answer, agent, search, harness_wall_s, transcript_rel, shim_log_rel = run_agent(
                instance, condition, tree, run_dir, args
            )
            answer_files, answer_functions = answer or ([], [])
            if args.no_score:
                # Capture runs (§16 C3): the agent never sees gold either way,
                # but a novel tool description must not mint accuracy-shaped
                # numbers as a side effect of harvesting queries.
                metrics = None
            else:
                metrics = scoring.score_instance(
                    answer_files, answer_functions, gold_files, gold_funcs,
                    worktree_prefix=str(tree),
                )
                metrics["first_hit_search_seq"] = scoring.first_gold_hit_seq(
                    DATA / shim_log_rel,
                    DATA / shim_log_rel.replace("shim_log.jsonl", "searches"),
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
            remove_worktree(repo, sha, iid)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def stratified_sample(rows, limit, seed):
    """Deterministic sample spread across categories, then repos within each.

    The seed has to actually move the sample. It used to barely: the previous
    version shuffled each category and then `sort(key=repo)`, and Python's sort
    is stable, so the shuffle survived only *within* a repo while the repo order
    came out alphabetical for every seed. Taking from the front then picked the
    same alphabetically-first repos every time — seed 1 and seed 2 shared 37 of
    40 instances. Anyone re-running under a new seed to get an independent
    sample would have got a near-duplicate with nothing saying so.

    Interleaving by repo is still the goal (one repo must not dominate a small
    sample), so the fix is to shuffle the repo *order* rather than to stop
    grouping: shuffle within each repo, shuffle the repos, then round-robin.
    """
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("category") or "?"].append(r)

    ordered = {}
    for cat, rs in by_cat.items():
        by_repo = defaultdict(list)
        for r in rs:
            by_repo[r["repo"]].append(r)
        for group in by_repo.values():
            rng.shuffle(group)
        repos = sorted(by_repo)            # sort first so the shuffle, not
        rng.shuffle(repos)                 # dict order, is what the seed drives
        # Round-robin across repos: spreads the sample without pinning it to
        # whichever repo sorts first.
        interleaved, i = [], 0
        while len(interleaved) < len(rs):
            for repo in repos:
                if i < len(by_repo[repo]):
                    interleaved.append(by_repo[repo][i])
            i += 1
        ordered[cat] = interleaved

    picked, cats = [], sorted(ordered)
    while len(picked) < limit and any(ordered[c] for c in cats):
        for c in cats:
            if ordered[c] and len(picked) < limit:
                picked.append(ordered[c].pop(0))
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
    ap.add_argument("--no-score", action="store_true",
                    help="capture-only: emit rows with metrics=None (§16 C3)")
    ap.add_argument("--evict-mirrors", action="store_true",
                    help="delete a repo's bare mirror once its last selected "
                         "instance completes — caps disk at ~one large repo "
                         "instead of the whole benchmark's mirror set (§16.9)")
    ap.add_argument("--max-new", type=int, default=0,
                    help="stop cleanly after N newly-emitted rows this "
                         "invocation (chunked runs across usage windows; "
                         "pair with --resume)")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="give up on an (instance, condition) after N failed "
                         "attempts so a chunked campaign can terminate")
    ap.add_argument("--disk-floor-gb", type=float, default=4.0,
                    help="refuse to start with less free disk than this "
                         "(0 disables); eval/reclaim.sh frees corpora/caches")
    ap.add_argument("--resume", action="store_true", help="skip (instance, condition, model) rows already ok in --out")
    ap.add_argument("--out", type=Path, default=DATA / "results.jsonl")
    args = ap.parse_args()

    if not SEMGREP.exists():
        sys.exit(f"build first: cargo build --release   (missing {SEMGREP})")
    if not Path(RG).exists():
        sys.exit(f"missing rg at {RG} (set RG_BIN)")
    if not args.dataset.exists():
        sys.exit(f"missing dataset: run  bash eval/locbench/fetch-dataset.sh")
    global CLAUDE_VERSION
    CLAUDE_VERSION = subprocess.run(
        [CLAUDE, "--version"], check=True, capture_output=True,
        text=True).stdout.strip()

    if args.disk_floor_gb:
        free_gb = os.statvfs(DATA).f_bavail * os.statvfs(DATA).f_frsize / 1e9
        if free_gb < args.disk_floor_gb:
            sys.exit(f"only {free_gb:.1f} GB free (< --disk-floor-gb "
                     f"{args.disk_floor_gb}); run eval/reclaim.sh or lower the "
                     f"floor. Mirrors for the full benchmark need ~8-15 GB "
                     f"unless --evict-mirrors is on.")

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines()]
    if args.instances:
        want = set(args.instances.split(","))
        rows = [r for r in rows if r["instance_id"] in want]
    else:
        rows = stratified_sample(rows, args.limit, args.seed)
    conditions = args.conditions.split(",")

    done = set()
    attempts = defaultdict(int)
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["instance_id"], r["condition"], r.get("model"))
            if r.get("status") == "ok":
                done.add(key)
            else:
                attempts[key] += 1
    # Abandonment (§16.9a D2): without this, a deterministically-failing
    # cell (dead repo, budget-blowing instance, model that won't emit JSON)
    # is retried at the head of every chunk forever and eats the --max-new
    # quota, so the campaign never terminates. Three strikes, then the cell
    # is treated as done and scored as a miss — pre-registered either way.
    abandoned = {k for k, n in attempts.items() if n >= args.max_attempts}
    if abandoned:
        print(f"abandoning {len(abandoned)} cells with >= {args.max_attempts} "
              f"failed attempts (scored as misses): "
              f"{sorted(k[0] for k in abandoned)[:5]}...", flush=True)
        done |= abandoned

    args.stop_event = None
    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = DATA / "runs" / args.run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "a")

    n_done = 0
    consecutive_errors = 0
    stop = threading.Event()
    outage = threading.Event()

    def emit(record):
        nonlocal n_done, consecutive_errors
        with emit_lock:
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_done += 1
            # Usage-outage guard (the generate.py lesson, §16.9): a wave of
            # instant agent_errors is a dead usage window, not data. Abort
            # loudly; every non-ok row re-runs under --resume.
            if record.get("status") in ("agent_error", "parse_error"):
                consecutive_errors += 1
                if consecutive_errors >= 6 and not stop.is_set():
                    stop.set()
                    outage.set()
                    print("STOPPING: 6 consecutive agent errors — the usage "
                          "window is likely exhausted. Re-run with --resume "
                          "when it resets; completed rows are kept.", flush=True)
            else:
                consecutive_errors = 0
            if args.max_new and n_done >= args.max_new and not stop.is_set():
                stop.set()
                print(f"STOPPING: --max-new {args.max_new} reached; "
                      f"re-run with --resume for the next chunk.", flush=True)
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

    # Rolling mirror eviction (§16.9): once the last selected instance of a
    # repo finishes in this process, its bare mirror is deleted. Keeps disk
    # bounded by the largest single repo; a later --resume re-clones.
    repo_remaining = defaultdict(int)
    for r, _ in todo:
        repo_remaining[r["repo"]] += 1
    evict_lock = threading.Lock()

    def work(t):
        r, conds = t
        if stop.is_set():
            return
        try:
            run_instance(r, conds, run_dir, args, emit)
        finally:
            if args.evict_mirrors:
                with evict_lock:
                    repo_remaining[r["repo"]] -= 1
                    evict = repo_remaining[r["repo"]] == 0
                if evict:
                    mirror = DATA / "repos" / "mirrors" / \
                        f"{r['repo'].replace('/', '__')}.git"
                    if mirror.exists():
                        with repo_locks[r["repo"]]:
                            # Any tree for this repo goes BEFORE its mirror:
                            # once the mirror is gone `git worktree remove` has
                            # nothing to operate on and the tree is stranded.
                            stray = sorted((DATA / "repos" / "trees").glob(
                                f"{r['repo'].replace('/', '__')}__*"))
                            for t in stray:
                                subprocess.run(["git", "-C", str(mirror), "worktree",
                                                "remove", "--force", str(t)],
                                               capture_output=True)
                                if t.exists():
                                    subprocess.run(["rm", "-rf", str(t)])
                            subprocess.run(["rm", "-rf", str(mirror)])
                        print(f"  evicted mirror {mirror.name}"
                              f"{f' (+{len(stray)} stray trees)' if stray else ''}",
                              flush=True)

    args.stop_event = stop
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    out_f.close()
    print(f"wrote {n_done} rows to {args.out}")
    if outage.is_set():
        # Distinct exit code so a `while :; do run.py --resume ...; done`
        # driver can tell "usage window died" from "chunk complete".
        sys.exit(3)


if __name__ == "__main__":
    main()
