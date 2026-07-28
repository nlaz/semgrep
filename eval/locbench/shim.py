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
import subprocess
import sys
import time
from pathlib import Path


def main():
    tool = sys.argv[1]
    argv = sys.argv[2:]
    real = os.environ.get(f"LOCBENCH_REAL_{tool.upper()}")
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
        out, err, code, wall_ms = b"", (msg + "\n").encode(), 2, 0.0
    else:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([real, *argv], capture_output=True, timeout=600)
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
        f.write(json.dumps({
            "seq": seq,
            "ts": time.time(),
            "tool": tool,
            "blocked": blocked,
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
