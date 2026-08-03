#!/usr/bin/env python3
"""Does the tool actually work on the input agents give it? (RESEARCH.md §16.11)

The §16.10 campaign spent $361 and 1,115 agent runs with 47% of the treatment
arm's searches silently returning nothing, because `semgrep "query" <file>`
was broken and nothing tested a file-as-root scope. Two adversarial reviews
missed it: they checked the experiment and the harness, not the tool's
behavior on real agent input.

This is that missing check. It replays **invocation shapes harvested from
real agent logs** (eval/queries/guesses-v0.jsonl — scope shape, flags, mode)
against a fixture corpus and fails if any of them comes back empty or
malformed. Run it before every campaign; it costs seconds and no API calls.

    python3 eval/locbench/preflight.py                 # all checks
    python3 eval/locbench/preflight.py --corpus PATH   # against another tree

Checks, each a hard failure:
  1. every real invocation SHAPE returns hits (dir / subdir / file scopes,
     ranked and exact, with and without -k)
  2. hits carry a non-empty path (the §16.11 sibling bug printed `:9:text`)
  3. no self-teaching footer leaks under SEMGREP_NO_HINTS (an A/B whose
     treatment arm withholds `-e` must not have the tool advertising it)
  4. grep muscle memory parses (§17): `-n` was 88% of the flags in the
     measured agent corpus and every one of those calls exited 2, as did the
     31% of calls passing more than one path
  5. the shim blocks what it should and passes through what it should
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SEMGREP = Path(os.environ.get("SEMGREP_BIN", ROOT / "target/release/semgrep"))
FIXTURE = ROOT / "tests/corpus"
GUESSES = HERE.parent / "queries" / "guesses-v0.jsonl"
SRC = re.compile(r"\.(py|rs|js|ts|go|java|rb|c|h|cpp|md|txt|json|ya?ml|toml|sh)$")

FAILURES = []


def fail(check, detail):
    FAILURES.append((check, detail))
    print(f"  FAIL  {check}: {detail}")


def ok(check, detail=""):
    print(f"  ok    {check}{(' — ' + detail) if detail else ''}")


def run(args, env_extra=None):
    env = dict(os.environ)
    env["SEMGREP_NO_HINTS"] = "1"
    env.update(env_extra or {})
    p = subprocess.run([str(SEMGREP), *args], capture_output=True, text=True,
                       timeout=120, env=env)
    hits = []
    for line in p.stdout.splitlines():
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return p, hits


def scope_shapes(corpus):
    """The scope shapes agents actually use, one representative of each."""
    src = sorted(p for p in corpus.rglob("*") if p.is_file() and SRC.search(p.name))
    subdirs = sorted({p.parent for p in src if p.parent != corpus})
    return [
        ("repo root", corpus),
        ("subdirectory", subdirs[0] if subdirs else corpus),
        # The shape that was broken for the entire life of the tool.
        ("single file", src[0] if src else corpus),
    ]


def literal_in(scope):
    """A literal that provably occurs in this scope, so an exact-mode check
    tests the TOOL rather than the author's guess about the fixture."""
    files = [scope] if scope.is_file() else [
        p for p in scope.rglob("*") if p.is_file() and SRC.search(p.name)]
    for f in files:
        try:
            for line in f.read_text(errors="replace").splitlines():
                for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", line):
                    return w
        except OSError:
            continue
    return "the"


def check_invocation_shapes(corpus, query):
    print("\n[1/5] real invocation shapes return hits")
    for label, scope in scope_shapes(corpus):
        lit = literal_in(scope)
        for mode_label, args in (("ranked", ["--json", query]),
                                 ("ranked -k 20", ["--json", "-k", "20", query]),
                                 (f"exact '{lit}'", ["--json", "-e", lit])):
            p, hits = run([*args, str(scope)])
            if p.returncode == 2:
                fail(f"{mode_label} @ {label}",
                     f"usage error (exit 2): {p.stderr.strip()[:80]!r} — the "
                     f"CHECK is malformed, not the tool")
                continue
            what = f"{mode_label} @ {label}"
            if not hits:
                fail(what, f"0 hits, exit {p.returncode} — a search that "
                           f"returns nothing while reporting success is the "
                           f"§16.11 failure mode")
            elif any(not h.get("path") for h in hits):
                fail(what, "a hit has an empty path")
            else:
                ok(what, f"{len(hits)} hits, first={hits[0]['path']}")


def check_real_guess_replay(corpus, n=25):
    """Replay actual logged agent invocations, shape-for-shape."""
    print(f"\n[2/5] replaying {n} real agent invocation shapes from the logs")
    if not GUESSES.exists():
        print(f"  skip  (no {GUESSES.name}; run locbench/harvest.py)")
        return
    rows = [json.loads(l) for l in GUESSES.read_text().splitlines()]
    ranked = [r for r in rows if r["kind"] == "guess_ranked"]
    # Sample across scope shapes, not just the head of the file.
    filey = [r for r in ranked if r["scopes_rel"] and SRC.search(r["scopes_rel"][0])]
    diry = [r for r in ranked if r["scopes_rel"] and not SRC.search(r["scopes_rel"][0])]
    sample = (filey[: n // 2] + diry[: n - n // 2]) or ranked[:n]
    src = sorted(p for p in corpus.rglob("*") if p.is_file() and SRC.search(p.name))
    empties = Counter()
    for r in sample:
        # Keep the agent's flags and scope SHAPE; retarget onto the fixture.
        scope = (src[0] if r["scopes_rel"] and SRC.search(r["scopes_rel"][0])
                 else corpus)
        args = ["--json"]
        if r.get("k"):
            args += ["-k", str(r["k"])]
        _, hits = run([*args, r["pattern"][:120], str(scope)])
        if not hits:
            empties["file" if scope != corpus else "dir"] += 1
    if empties:
        fail("guess replay",
             f"{sum(empties.values())}/{len(sample)} real invocation shapes "
             f"returned nothing ({dict(empties)}) — the tool does not answer "
             f"what agents type")
    else:
        ok("guess replay", f"all {len(sample)} shapes returned hits")


def check_no_coaching(corpus):
    """Two separate contracts, both shipped as of RESEARCH.md §16.10.

    (a) SEMGREP_NO_HINTS silences the footer entirely — the harness knob.
    (b) Ranked footers never name `-e` *even unsuppressed* — the product
        posture. Suppression alone was not enough: an arm whose description
        withholds `-e` is coached anyway the moment someone runs the binary
        without the env var, and the footer is the stronger dose of the two.
    """
    print("\n[3/5] footer carries no exact-mode coaching")
    p, _ = run(["--json", "-k", "3", "some query that will not match anything xyzzy",
                str(corpus)])
    if p.stderr.strip():
        fail("footer suppression",
             f"stderr not silent under SEMGREP_NO_HINTS: {p.stderr.strip()[:90]!r}")
    else:
        ok("footer suppression", "stderr silent")

    # Unsuppressed: the footer must exist (so the check above isn't vacuous)
    # and must still not advertise exact mode.
    env = dict(os.environ)
    env.pop("SEMGREP_NO_HINTS", None)
    q = subprocess.run([str(SEMGREP), "-k", "3", "retry backoff", str(corpus)],
                       capture_output=True, text=True, timeout=120, env=env)
    if "semgrep:" not in q.stderr:
        fail("footer suppression control",
             "no footer even WITHOUT the env var — the check above proves nothing")
    else:
        ok("footer suppression control", "footer present when not suppressed")
    if re.search(r"(?<![\w-])-e(?![\w-])|--exact", q.stderr):
        fail("ranked footer posture",
             f"the unsuppressed ranked footer names exact mode: "
             f"{q.stderr.strip()[:90]!r} — §16.10 measured this one line moving "
             f"ranked share 98% -> 7%")
    else:
        ok("ranked footer posture", "ranked footer does not advertise -e")


def check_grep_compat(corpus):
    """Does the tool accept what grep muscle memory types? (RESEARCH.md §17)

    Measured on the rg arm's 2,074 real invocations: `-n` appeared 1,829 times
    (88%), `-A`/`-B` 142, `-g`/`--include` 273, `-l` 118, `-r`/`-R` 48, and 31%
    of calls passed two or more paths. Every one of those exited 2. semgrep's
    `path:line:text` output *is* grep's `-rn` form, so accepting the no-ops
    costs nothing and refusing them cost a whole class of dead calls.

    Asserted as exit != 2 — a usage error. Zero hits is fine; being unable to
    parse the command line is not.
    """
    print("\n[4/5] grep muscle memory is accepted")
    src = sorted(p for p in corpus.rglob("*") if p.is_file() and SRC.search(p.name))
    subdirs = sorted({p.parent for p in src if p.parent != corpus})
    two = [str(subdirs[0]), str(subdirs[-1])] if len(subdirs) > 1 else [str(corpus)]
    cases = [
        ("-n (88% of observed flags)", ["-n", "-k", "3", "retry backoff"]),
        ("-r", ["-r", "-k", "3", "retry backoff"]),
        ("-R", ["-R", "-k", "3", "retry backoff"]),
        ("-H", ["-H", "-k", "3", "retry backoff"]),
        ("-rn combined", ["-rn", "-k", "3", "retry backoff"]),
        ("-rln combined", ["-rln", "-k", "3", "retry backoff"]),
        ("-A 3", ["-A", "3", "-k", "2", "retry backoff"]),
        ("-B 3", ["-B", "3", "-k", "2", "retry backoff"]),
        ("-l", ["-l", "-k", "3", "retry backoff"]),
        ("-g glob", ["-g", "*.md", "-k", "3", "retry backoff"]),
        ("--include glob", ["--include", "*.md", "-k", "3", "retry backoff"]),
    ]
    for label, args in cases:
        p, _ = run([*args, str(corpus)])
        if p.returncode == 2:
            fail(f"compat {label}", f"exit 2: {p.stderr.strip()[:70]!r}")
        else:
            ok(f"compat {label}", f"exit {p.returncode}")
    # Multiple paths: grep's contract, 31% of real invocations.
    p, _ = run(["-k", "3", "retry backoff", *two])
    if p.returncode == 2:
        fail("compat multi-path", f"exit 2: {p.stderr.strip()[:70]!r}")
    else:
        ok("compat multi-path", f"exit {p.returncode} over {len(two)} paths")
    # The control. If an unknown flag stopped failing, the checks above would
    # pass because argument parsing had gone permissive, not because the flags
    # are handled — and the whole section would be measuring nothing.
    p, _ = run(["--definitely-not-a-flag", "-k", "3", "retry backoff", str(corpus)])
    if p.returncode != 2:
        fail("compat control", "an unknown flag no longer errors — the checks "
                               "above prove nothing")
    else:
        ok("compat control", "an unknown flag still exits 2")


def check_shims():
    print("\n[5/5] shim blocks and passes through as configured")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sys.path.insert(0, str(HERE))
        import run as locbench
        (td / "bin").mkdir()
        locbench.make_shims(td / "bin")
        env = dict(os.environ)
        env["PATH"] = f"{td / 'bin'}:{env['PATH']}"
        env["LOCBENCH_SHIM_LOG"] = str(td / "log.jsonl")
        env["LOCBENCH_STDOUT_DIR"] = str(td / "out")
        env.update(locbench.block_msgs("desc-v5"))
        for tool, should_work in (("grep", False), ("git", False), ("rg", False)):
            p = subprocess.run([tool, "--version"], capture_output=True, text=True,
                               env=env, timeout=60)
            blocked = p.returncode == 2 and not p.stdout
            if blocked != (not should_work):
                fail(f"shim {tool}", f"expected blocked={not should_work}, "
                                     f"got exit {p.returncode}")
            else:
                ok(f"shim {tool}", "blocked" if blocked else "passes through")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=FIXTURE)
    ap.add_argument("--query", default="compute the retry backoff delay")
    ap.add_argument("--skip-shims", action="store_true")
    args = ap.parse_args()

    if not SEMGREP.exists():
        raise SystemExit(f"build first: cargo build --release (missing {SEMGREP})")
    print(f"preflight against {args.corpus} using {SEMGREP}")

    check_invocation_shapes(args.corpus, args.query)
    check_real_guess_replay(args.corpus)
    check_no_coaching(args.corpus)
    check_grep_compat(args.corpus)
    if not args.skip_shims:
        check_shims()

    print()
    if FAILURES:
        print(f"PREFLIGHT FAILED — {len(FAILURES)} check(s). Do not spend money "
              f"on a campaign until these pass:")
        for c, d in FAILURES:
            print(f"  · {c}: {d}")
        sys.exit(1)
    print("preflight passed — the tool answers what agents actually type.")


if __name__ == "__main__":
    main()
