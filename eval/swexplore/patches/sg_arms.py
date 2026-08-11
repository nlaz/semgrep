"""The three §27 arms, as SWE-Explore explorers.

Dropped into the upstream tree at `explorers/sg_arms.py` by `fetch.sh`.
Subclasses their `ClaudeCodeExplorer` so `EXPLORE_PROMPT` and
`parse_relevant_files` are inherited **verbatim** — the task contract and the
output parser are theirs, which is what keeps `cc` comparable to the row
published in arXiv 2606.07297 (HitReg 0.531 / HitFile 0.667 / CtxEff 0.829).

    cc      Read,Glob,Grep                  their baseline
    cc-rg   + Bash(rg *)                    Bash-balanced control
    cc-sg   + Bash(sg *)                    treatment

Three contrasts: cc→cc-sg is the product question, cc-rg→cc-sg isolates
semgrep from Bash, cc→cc-rg prices the Bash confound itself.

What this adds on top of their explorer, and nothing else:

  * `--output-format stream-json --verbose`, teed to `transcript.jsonl`.
    Theirs uses `--output-format json`, which returns one blob: no per-turn
    usage, no cost, no tool-call sequence. Cost is the co-primary endpoint
    here, so the stream is not optional.
  * `telemetry.jsonl` — `total_cost_usd`, `num_turns`, `usage`, `duration_ms`
    off the terminal `{"type":"result"}` event. Same fields and the same
    keep-`usage`-whole convention as `locbench/run.py:696-703`, so a new
    usage field survives without a harness change.
  * `locbench/shim.py` on PATH, unchanged — it derives the real binary from
    `argv[1]`, so `sg` needs no edit there. Buys per-invocation argv, exit,
    stdout bytes and wall ms, which is what the sg-invocation tripwire reads.
  * `SEMGREP_CACHE_DIR` per arm and `SEMGREP_TRACE_FILE` per cond dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import ExplorerResult
from .claude_code import ClaudeCodeExplorer

# --------------------------------------------------------------------------
# Tool lines
# --------------------------------------------------------------------------
# These are NOT desc-v9 verbatim, and the plan said they would be. Two things
# forced the change, both of which would have made the description false:
#
#   1. desc-v9 opens "The only code search tool available is `sg`" and closes
#      "Read and Glob are also available." Both are wrong in an *additive*
#      arm, where Grep and Glob remain. Telling an agent it has no grep when
#      it does is not a description variant, it is a lie the agent can check.
#   2. desc-v9 says "top 10". The shipped default has been `k: 5` since §26.3.
#
# So the availability framing and the count changed; the mechanism sentence,
# the example, and the "ranked, not exhaustive" caveat are desc-v9's own words
# and are asserted against it by `tools_lines_track_descv9` below.
#
# The rg line is deliberately built to the SAME shape — opener, mechanism,
# worked example — because §7.3 and §19 both measured that description quality
# moves agent behaviour. An rg arm described worse than the sg arm would turn
# the cc-rg→cc-sg contrast into a description contrast. locbench's own `rg`
# line has no example, which is exactly why CLAUDE.md keeps a separate
# `rg-strong` condition; this is the strong form.

RG_LINE = (
    "Additionally, `rg` (ripgrep) is available via Bash: `rg [flags] <regex> "
    "[path]` for exact or regex content matching, printing every match as "
    "path:line:text. "
    'Example: rg "fn backoff_delay" src/ → '
    "src/net/retry.rs:142:fn backoff_delay(attempt: u32). "
    "Exhaustive, not ranked — if it floods, narrow the path or the pattern."
)

SG_LINE = (
    "Additionally, `sg` is available via Bash, a ranked code search. "
    "Give it anything — an identifier, a phrase, or a question: "
    '`sg "query" [path]` returns the most relevant locations as '
    "path:line:text (top 5; `-k N` for more). "
    'Example: sg "retry_backoff backoff_delay compute_delay" → '
    "src/net/retry.rs:142:fn backoff_delay(attempt: u32). "
    "Ranked, not exhaustive — if the answer isn't there, rephrase."
)

# The desc-v10 framing (wide search as the default, the path as a deliberate
# second step — §28.2's "gold scoped away" bucket is the target). NOT wired
# into any arm: §28's registered arms ran with SG_LINE above, and the 364
# rate-limited sub-sg cells must complete under the registered treatment or
# the arm becomes a mixture. A future campaign registers arms on this line.
SG_LINE_V10 = (
    "Additionally, `sg` is available via Bash, a ranked code search. "
    "Give it anything — an identifier, a phrase, or a question: "
    '`sg "query"` searches the whole repository and returns the most '
    "relevant locations as path:line:text (top 5; `-k N` for more). "
    "Start wide: add a path argument only to narrow further after a wide "
    "search has pointed somewhere. "
    'Example: sg "retry_backoff backoff_delay compute_delay" → '
    "src/net/retry.rs:142:fn backoff_delay(attempt: u32). "
    "Ranked, not exhaustive — if the answer isn't there, rephrase."
)

# arm -> (tools surface, auto-approve allowlist, appended system prompt)
#
# §27 arms are ADDITIVE: native Grep stays and a Bash tool is offered beside it.
# §28 `sub-*` arms are SUBSTITUTIVE: Grep is removed from `--tools` entirely, so
# the Bash tool is the only content search available. That is the point — §27
# measured 41% delivery because the agent kept choosing Grep, and three separate
# measurements now show it always will when a lexical tool is present.
#
# Removal is enforced in two places, and both are needed: dropping `Grep` from
# `--tools` takes away the native tool, and the PATH shims block shell
# `grep`/`egrep`/`fgrep`. `--allowedTools` does NOT enforce anything under
# `--permission-mode dontAsk` (see _make_shims).
ARMS = {
    "cc":     ("Read,Glob,Grep",      [],             ""),
    "cc-rg":  ("Read,Glob,Grep,Bash", ["Bash(rg *)"], RG_LINE),
    "cc-sg":  ("Read,Glob,Grep,Bash", ["Bash(sg *)"], SG_LINE),
    "sub-rg": ("Read,Glob,Bash",      ["Bash(rg *)"], RG_LINE),
    "sub-sg": ("Read,Glob,Bash",      ["Bash(sg *)"], SG_LINE),
}
ARM_TOOL = {"cc-rg": "rg", "cc-sg": "sg", "sub-rg": "rg", "sub-sg": "sg"}

# --------------------------------------------------------------------------
# The one clause of upstream's prompt we rewrite, and why
# --------------------------------------------------------------------------
# Their EXPLORE_PROMPT instructs "Use Glob, Grep, and Read tools to explore the
# codebase." The first smoke measured the consequence: across every arm and
# every instance, `bash_calls` was **0** — neither `sg` nor `rg` was invoked
# once, all three arms returned identical answers, and the treatment was
# simply never delivered. An appended system prompt saying the tool exists
# does not survive a user prompt naming three others; §25's "availability is
# not use", in a new place.
#
# So the clause is amended for the two treatment arms and ONLY that clause:
# same position, same shape, one tool name added. `cc` keeps upstream's prompt
# byte-for-byte, which is what preserves it as the calibration anchor against
# the published row.
#
# `_amend` asserts the clause is present. If a future upstream edits the
# wording, the arm dies loudly rather than quietly reverting to a no-op
# treatment that would read as a clean null.
PROMPT_CLAUSE = "Use Glob, Grep, and Read tools to explore the codebase."
ARM_CLAUSE = {
    "cc": None,
    "cc-rg": "Use Glob, Grep, Read, and the `rg` command (via Bash) to explore the codebase.",
    "cc-sg": "Use Glob, Grep, Read, and the `sg` command (via Bash) to explore the codebase.",
    # The sub-* clauses drop Grep, because for these arms it does not exist and
    # naming it would send the agent at a tool it cannot call. Otherwise the
    # sentence keeps its position and shape, so the only difference between
    # cc-sg and sub-sg is the presence of Grep — which is the treatment.
    "sub-rg": "Use Glob, Read, and the `rg` command (via Bash) to explore the codebase.",
    "sub-sg": "Use Glob, Read, and the `sg` command (via Bash) to explore the codebase.",
}


def _amend(prompt: str, arm: str) -> str:
    clause = ARM_CLAUSE[arm]
    if clause is None:
        return prompt
    if PROMPT_CLAUSE not in prompt:
        raise RuntimeError(
            f"{arm}: upstream EXPLORE_PROMPT no longer contains the clause this "
            f"arm rewrites ({PROMPT_CLAUSE!r}). Re-register the arm rather than "
            f"running a treatment that is never delivered.")
    return prompt.replace(PROMPT_CLAUSE, clause)

HERE = Path(__file__).resolve().parent
# eval/data/swexplore/upstream/explorers -> repo root
REPO_ROOT = HERE.parents[4]
LOCBENCH = REPO_ROOT / "eval" / "locbench"
SEMGREP_BIN = Path(os.environ.get("SEMGREP_BIN", REPO_ROOT / "target/release/sg"))
RG_BIN = os.environ.get("RG_BIN", "/opt/homebrew/bin/rg")

# Provenance level. `full` is for the ladder rungs, where the point is to be
# able to read what happened; `lean` is for the powered run, where 2,544 cells
# of transcripts and search dumps is ~650 MB nobody will open.
#
# The switch is safe to flip mid-campaign, and that is a load-bearing claim
# rather than a convenience: every artifact it drops is written *after* the
# bytes have already been replayed to the agent, so agent-visible behaviour is
# identical and lean rungs pool with full ones. What it must NOT touch:
#   * the shim itself and its block messages — that is the mechanism that
#     keeps shell `grep` away from the arms, not logging;
#   * `SEMGREP_NO_HINTS` — §16.10 measured that footer moving an agent's
#     ranked share from 7% to 98%, so it is a treatment;
#   * `--stats` / `--stats-json` — those write to stderr, which shim.py
#     replays to the agent, so they are agent-visible by construction and are
#     never used here at any level.
PROV = os.environ.get("SWEXPLORE_PROV", "full")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_BIN_SHA: str | None = None


def _binary_sha() -> str | None:
    """sha256 of the sg binary, computed once. 39 MB × 2,544 runs is 4 minutes
    of hashing for a constant, so it is cached rather than recomputed."""
    global _BIN_SHA
    if _BIN_SHA is None and SEMGREP_BIN.exists():
        h = hashlib.sha256()
        with SEMGREP_BIN.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _BIN_SHA = h.hexdigest()
    return _BIN_SHA


def tools_lines_track_descv9() -> dict:
    """Assert our sg line still carries desc-v9's own mechanism sentences.

    Run by `preflight.py`. The two clauses below are the parts §19 measured;
    if a future desc-vN edits them, this fires and the arm is re-registered
    deliberately rather than drifting into a different treatment silently.
    """
    sys.path.insert(0, str(LOCBENCH))
    import run as locbench  # noqa: E402

    v9 = locbench.TOOL_LINES["desc-v9"]
    shared = [
        "Give it anything — an identifier, a phrase, or a question",
        'Example: sg "retry_backoff backoff_delay compute_delay"',
        "Ranked, not exhaustive — if the answer isn't there, rephrase.",
    ]
    missing = [s for s in shared if s not in v9 or s not in SG_LINE]
    return {"ok": not missing, "missing": missing,
            "descv9_sha256": _sha(v9), "sg_line_sha256": _sha(SG_LINE)}


@dataclass
class ArmExplorer(ClaudeCodeExplorer):
    """One §27 arm. `repo_root`, `model`, `timeout` come from the base class."""

    arm: str = "cc"
    run_dir: Path = field(default_factory=lambda: Path("runs/adhoc"))
    # Their explorer hardcodes bypassPermissions. That skips permission checks
    # entirely, which makes --allowed-tools advisory rather than binding — so
    # `cc` could shell out and the confound design would be void. `dontAsk` is
    # what locbench has used for every campaign since §16 and it binds the
    # allowlist. Overridable so the two can be compared head to head, which
    # preflight does on one instance before anything is spent.
    permission_mode: str = "dontAsk"

    # ---------------------------------------------------------------- env
    def _cond_dir(self, instance_id: str) -> Path:
        d = self.run_dir / instance_id / self.arm
        (d / "searches").mkdir(parents=True, exist_ok=True)
        return d

    def _make_shims(self, bin_dir: Path) -> None:
        """One wrapper per shimmed *and* blocked name, front of PATH.
        Mirrors run.py:485-494.

        BLOCKED is load-bearing and was learned the hard way. `--allowedTools
        Bash(rg *)` does NOT bind under `--permission-mode dontAsk` — dontAsk
        means "do not prompt", not "restrict". The first amended smoke caught
        the consequence: cc-rg ran `grep -n "n_jobs" sklearn/...` straight
        through the Bash tool and never touched rg at all. Left in, every
        Bash-enabled arm can do lexical search without invoking its own
        treatment, all three arms converge on shell grep, and the campaign
        reports a clean null that means nothing.

        shim.py blocks any name with no LOCBENCH_REAL_* binding, so listing
        them here is the whole mechanism. `git` goes too, for the reason
        run.py:477-482 gives: history after base_commit contains the real
        fix, and `git log --all --grep=<issue>` is ground truth.
        """
        bin_dir.mkdir(parents=True, exist_ok=True)
        for tool in ("rg", "sg", "grep", "egrep", "fgrep", "git"):
            w = bin_dir / tool
            w.write_text(
                f'#!/bin/sh\nexec /usr/bin/env python3 "{LOCBENCH / "shim.py"}" {tool} "$@"\n'
            )
            w.chmod(0o755)

    def _env(self, cond_dir: Path) -> tuple[dict, str]:
        bin_dir = cond_dir / "bin"
        self._make_shims(bin_dir)
        path = f"{bin_dir}:{os.environ['PATH']}"
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        env.update({
            "PATH": path,
            "LOCBENCH_SHIM_LOG": str(cond_dir / "shim_log.jsonl"),
            "LOCBENCH_STDOUT_DIR": str(cond_dir / "searches"),
            # Sibling of runs/, not inside it. Under run_dir it created
            # runs/<run>/semgrep-cache/<arm>/, which sits in the same namespace
            # as the instance directories — so anything globbing
            # runs/<run>/*/ picks up a cache dir as if it were an instance.
            "SEMGREP_CACHE_DIR": str(self.run_dir.parent.parent / "cache"
                                     / self.run_dir.name / self.arm),
            "SEMGREP_NO_HINTS": "1",
            "SEMGREP_TRACE_FILE": str(cond_dir / "trace.jsonl"),
            "LOCBENCH_STDOUT_DUMP": "on" if PROV == "full" else "off",
        })
        # Only this arm's own tool gets a real binding; the other falls into
        # shim.py's blocked path, so an arm escaping its treatment is visible
        # as a blocked row rather than silently succeeding.
        for t in ("rg", "sg"):
            env.pop(f"LOCBENCH_REAL_{t.upper()}", None)
            env.pop(f"LOCBENCH_{t.upper()}_FLAGS", None)
        tool = ARM_TOOL.get(self.arm)
        if tool == "rg":
            env["LOCBENCH_REAL_RG"] = RG_BIN
        elif tool == "sg":
            env["LOCBENCH_REAL_SG"] = str(SEMGREP_BIN)
        # Steer rather than just refuse: shim.py writes these on both stdout
        # and stderr, because an agent piping into `head` sees silence
        # otherwise and reads it as "no matches" (run.py:514-531).
        steer = (f"use the {tool} command instead" if tool
                 else "use the Grep and Glob tools instead")
        for t in ("grep", "egrep", "fgrep", "rg", "sg"):
            env[f"LOCBENCH_BLOCKMSG_{t.upper()}"] = (
                f"{t}: unavailable in this environment — {steer}")
        env["LOCBENCH_BLOCKMSG_GIT"] = (
            "git: unavailable in this environment — do not retry git commands; "
            "search the working tree instead")
        return env, path

    def _ensure_index(self) -> dict:
        """Build .semgrep in the checkout. Only the sg arm gets one."""
        if ARM_TOOL.get(self.arm) != "sg":
            return {"built": False, "reason": "arm has no sg"}
        idx = Path(self.repo_root) / ".semgrep" / "meta.json"
        if idx.exists() and idx.stat().st_mtime >= SEMGREP_BIN.stat().st_mtime:
            return {"built": False, "reason": "reused"}
        t0 = time.time()
        p = subprocess.run([str(SEMGREP_BIN), "index", str(self.repo_root)],
                           capture_output=True, timeout=3600)
        return {"built": p.returncode == 0, "index_s": round(time.time() - t0, 2),
                "returncode": p.returncode,
                "stderr": p.stderr.decode(errors="ignore")[-400:] if p.returncode else ""}

    # ------------------------------------------------------------- explore
    def explore(self, *, instance_id: str, query: str, top_k: int = 5
                ) -> List[ExplorerResult]:
        from .claude_code import EXPLORE_PROMPT  # their prompt, untouched
        from .parsing import parse_relevant_files

        cond_dir = self._cond_dir(instance_id)
        tools, allowed, sysline = ARMS[self.arm]
        prompt = _amend(EXPLORE_PROMPT.format(issue=query, top_k=top_k), self.arm)
        index = self._ensure_index()
        env, path = self._env(cond_dir)

        cmd = [
            "claude", "-p",
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", self.permission_mode,
            "--model", self.model,
            "--tools", tools,
            "--no-session-persistence",
            # The first smoke exposed 36 tools, 32 of them MCP servers from
            # the operator's own config (Google Drive, Gmail, Playwright).
            # That is contamination three ways: capabilities the benchmark
            # never granted, a system prompt inflated by 32 tool schemas —
            # which is cache-creation tokens, i.e. the co-primary endpoint —
            # and a configuration nobody else could reproduce. Both flags,
            # because --strict-mcp-config alone still lets settings files in.
            "--strict-mcp-config",
            "--setting-sources", "",
            # Belt and braces against a login shell resetting PATH, exactly
            # as run.py:652 does — without it the shims are bypassed and the
            # agent talks to the real binary with no log.
            "--settings", json.dumps({"env": {"PATH": path}}),
        ]
        if allowed:
            cmd += ["--allowedTools", *allowed]
        if sysline:
            cmd += ["--append-system-prompt", sysline]

        (cond_dir / "meta.json").write_text(json.dumps({
            "instance_id": instance_id, "arm": self.arm, "model": self.model,
            "tools": tools, "allowed_tools": allowed,
            "permission_mode": self.permission_mode,
            "provenance": PROV,
            # Both recorded: the amended prompt this arm actually sent, and
            # upstream's unamended one. cc's two are equal by construction, so
            # the pair is the audit trail for exactly what diverged and where.
            "user_prompt_sha256": _sha(prompt),
            "upstream_prompt_sha256": _sha(EXPLORE_PROMPT.format(issue=query, top_k=top_k)),
            "system_prompt_sha256": _sha(sysline) if sysline else None,
            "semgrep_sha256": _binary_sha(),
            "index": index, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=1) + "\n")

        transcript = cond_dir / "transcript.jsonl"
        status, t0 = "ok", time.time()
        # Binary mode throughout: the transcript is streamed straight to disk
        # so a mid-run crash still leaves everything up to that point, and the
        # prompt goes in as bytes to match. Theirs buffers stdout in memory
        # and loses it on a timeout.
        with open(transcript, "wb") as tf:
            try:
                p = subprocess.run(cmd, cwd=str(self.repo_root),
                                   input=prompt.encode(),
                                   stdout=tf, stderr=subprocess.PIPE,
                                   timeout=self.timeout, env=env)
                if p.returncode != 0:
                    status = "agent_error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        wall = round(time.time() - t0, 2)

        result_text, agent = "", {"status": status, "harness_wall_s": wall}
        bash_calls = grep_calls = 0
        for line in transcript.read_text(errors="ignore").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # `message` is sometimes a bare string and `content` sometimes a
            # bare string — the same shape that crashed displaychmp.walk()
            # (§25.3). Guard both, or a single such event kills the arm after
            # the money has already been spent.
            msg = ev.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") == "Bash":
                        bash_calls += 1
                    elif block.get("name") == "Grep":
                        grep_calls += 1
            if ev.get("type") == "result":
                agent.update(
                    num_turns=ev.get("num_turns"),
                    total_cost_usd=ev.get("total_cost_usd"),
                    duration_ms=ev.get("duration_ms"),
                    duration_api_ms=ev.get("duration_api_ms"),
                    usage=ev.get("usage") or {},   # whole and unparsed
                )
                result_text = ev.get("result") or ""
                if ev.get("is_error") and agent["status"] == "ok":
                    agent["status"] = "agent_error"

        shim = self._read_shim_log(cond_dir)
        agent.update(bash_calls=bash_calls, grep_calls=grep_calls, **shim)
        agent.update(instance_id=instance_id, arm=self.arm, index=index)
        # One file per cell rather than a shared append log: the runner joins
        # this back by (instance_id, arm) and a per-cell file needs no lock,
        # no scan, and is overwritten cleanly by a --resume retry.
        (cond_dir / "telemetry.json").write_text(json.dumps(agent, indent=1) + "\n")

        # The transcript is *written* at both levels — it is streamed to disk so
        # a crash still leaves everything up to that point, and the cost fields
        # above are parsed out of it. Under `lean` it is discarded once parsed,
        # which is why the endpoint survives losing the file.
        if PROV != "full":
            transcript.unlink(missing_ok=True)

        return parse_relevant_files(result_text, instance_id, top_k=top_k)

    @staticmethod
    def _read_shim_log(cond_dir: Path) -> dict:
        log = cond_dir / "shim_log.jsonl"
        rows = []
        if log.exists():
            for line in log.read_text(errors="ignore").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        live = [r for r in rows if not r.get("blocked")]
        return {
            "n_invocations": len(live),
            "n_blocked": sum(1 for r in rows if r.get("blocked")),
            "n_by_tool": {t: sum(1 for r in live if r.get("tool") == t)
                          for t in ("rg", "sg")},
            "total_stdout_bytes": sum(r.get("stdout_bytes") or 0 for r in live),
            "total_search_ms": round(sum(r.get("wall_ms") or 0 for r in live), 1),
            "n_nonzero_exit": sum(1 for r in live if r.get("exit")),
        }
