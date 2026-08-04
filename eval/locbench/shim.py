#!/usr/bin/env python3
"""Logging shim for search tools in Loc-Bench agent runs.

Invoked by the generated `rg`/`semgrep` wrappers on the agent's PATH:

    shim.py <tool> [args...]

Runs the real binary, passes stdout/stderr/exit through byte-exact (the
agent must see zero behavioral difference — no flag injection), and records
per-invocation provenance to LOCBENCH_SHIM_LOG (JSONL) plus the full stdout
to LOCBENCH_STDOUT_DIR/NNN.out for post-hoc searches-to-first-gold-hit
scoring and offline --stats replay.

Env contract (set by run.py):
    LOCBENCH_REAL_RG / LOCBENCH_REAL_SEMGREP  absolute real binaries
    LOCBENCH_SHIM_LOG                          JSONL log path
    LOCBENCH_STDOUT_DIR                        dir for captured stdouts
"""

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    tool = sys.argv[1]
    argv = sys.argv[2:]
    real = os.environ.get(f"LOCBENCH_REAL_{tool.upper()}")
    # Per-condition engine flags (A/B): appended to the real invocation but
    # never shown to the agent — its commands and the logged argv stay clean.
    injected = ""
    if real is not None and tool in ("semgrep", "search"):
        injected = os.environ.get("LOCBENCH_SEMGREP_FLAGS", "")
    log_path = Path(os.environ["LOCBENCH_SHIM_LOG"])
    stdout_dir = Path(os.environ["LOCBENCH_STDOUT_DIR"])

    blocked = real is None
    if blocked:
        # Condition-isolation blocker: Claude Code auto-allows read-only Bash
        # commands, so grep/git can't be excluded via --allowedTools — they
        # are intercepted here instead, with a steer to the condition's tool.
        msg = os.environ.get(
            f"LOCBENCH_BLOCKMSG_{tool.upper()}",
            f"{tool} is unavailable in this environment",
        )
        # On BOTH streams, because this is an instruction rather than an error
        # and it must reach the agent however it wrote the command. Agents pipe
        # (`grep -l … 2>/dev/null | head`), and stderr-only meant the refusal
        # was discarded and the agent saw an empty result with no reason for
        # it — silence that looks like a real answer, which is the §16.11
        # failure mode this harness exists to catch. Observed live: one tier-1b
        # agent redirected stderr and got "(Bash completed with no output)".
        #
        # Safe on stdout because the message names the tool and says what to do
        # instead, so it cannot be mistaken for search results, and because
        # every metric in run.py's search stats skips blocked rows before
        # counting bytes.
        payload = (msg + "\n").encode()
        out, err, code, wall_ms = payload, payload, 2, 0.0
    else:
        t0 = time.perf_counter()
        run_argv = argv + shlex.split(injected) if injected else argv
        try:
            proc = subprocess.run([real, *run_argv], capture_output=True, timeout=600)
            out, err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            out, err, code = e.stdout or b"", (e.stderr or b"") + b"\nshim: timeout after 600s\n", 124
        wall_ms = (time.perf_counter() - t0) * 1e3

    # Log first (under flock: the agent may run searches concurrently), then
    # replay the output so the agent-visible behavior is byte-identical.
    stdout_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        seq = sum(1 for _ in f)
        stdout_file = stdout_dir / f"{seq:03d}.out"
        stdout_file.write_bytes(out)
        # Everything the agent was TOLD, not just how many bytes of it:
        # semgrep's footers coach mode choice on stderr, which is a
        # treatment channel in an A/B (§16.9a A1). Only stderr_bytes
        # survived before, so the channel was unauditable after the fact.
        if err:
            (stdout_dir / f"{seq:03d}.err").write_bytes(err)
        f.write(json.dumps({
            "seq": seq,
            "ts": time.time(),
            "tool": tool,
            "blocked": blocked,
            "injected": injected,
            "argv": argv,
            "cwd": os.getcwd(),
            "wall_ms": round(wall_ms, 1),
            "exit": code,
            "stdout_bytes": len(out),
            "stderr_bytes": len(err),
            "stdout_file": stdout_file.name,
        }) + "\n")
        f.flush()

    sys.stdout.buffer.write(out)
    sys.stderr.buffer.write(err)
    sys.exit(code)


if __name__ == "__main__":
    main()
