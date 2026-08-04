#!/usr/bin/env python3
"""Query shape by condition — the §19 description A/B's first endpoint.

§19.3 registers query *style* as prediction 1, and registers it first on
purpose: it is the mechanism a micro-example acts through, it is readable from
the shim logs with no scoring and no gold files, and it gates the rest. If the
description does not change how agents write queries, the accuracy predictions
are void rather than negative.

Style, not length. §19.2b measured that length is the wrong proxy — a
paraphrase is *long and bad*, finding the gold 4% of the time when it shares
no vocabulary with it, against 46% for an equally-blind identifier guess. So
what a description has to move is the shape of the query, and a length that
rose because agents started writing questions would be a regression reported
as a win.

Reads `harvest.py`'s rows rather than the shim logs directly: argv parsing for
a tool whose flags keep changing belongs in one place, and harvest already has
the reconciliation gate that catches a silent parse failure.

    python3 eval/locbench/queryshape.py                        # every condition
    python3 eval/locbench/queryshape.py --a desc-v8 --b desc-v5
    python3 eval/locbench/queryshape.py --runs ../data/locbench/runs/20260804-1200

Ranked queries only. Exact-mode patterns are regexes, where "words" is not a
meaningful unit and a long pattern means the opposite of what it means in a
ranked query.
"""

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harvest import harvest  # noqa: E402

DATA = HERE.parent / "data" / "locbench"

# English function words. Their presence is what makes a query a description
# rather than a name — and SIF weights them toward zero (a/(a+p)), so they are
# close to free at the engine and decisive only as a classification signal.
FUNC = {"where", "is", "the", "a", "an", "how", "what", "does", "do", "in",
        "to", "for", "of", "and", "are", "when", "which", "that", "this", "it",
        "on", "with", "by", "from", "be", "was", "were", "or", "if", "then",
        "we", "you", "i", "its", "their", "there", "here", "why", "who", "can",
        "should", "would", "get", "set"}
SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")
CAMEL = re.compile(r"^[a-z]+([A-Z][a-z0-9]*)+$|^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)+$")
DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")

STYLES = ["identifiers", "plain words", "mixed", "paraphrase"]


def code_token(tok):
    tok = tok.strip("(),:\"'`")
    return bool(SNAKE.match(tok) or CAMEL.match(tok) or DOTTED.match(tok))


def classify(query):
    """Which of the four shapes §19.2b scored this query as.

    `identifiers` is a name or several candidate names; `paraphrase` is a
    description in English; `plain words` is content words with neither code
    casing nor function words ("retry backoff"); `mixed` is both.
    """
    toks = query.split()
    if not toks:
        return None
    code = sum(1 for t in toks if code_token(t))
    func = sum(1 for t in toks if t.lower().strip(".,?") in FUNC)
    if code and not func:
        return "identifiers"
    if func and not code:
        return "paraphrase"
    if code and func:
        return "mixed"
    return "plain words"


def summarize(queries):
    n = len(queries)
    if not n:
        return None
    lengths = [len(q.split()) for q in queries]
    styles = Counter(classify(q) for q in queries)
    return {
        "n": n,
        "mean_words": statistics.mean(lengths),
        "styles": {s: styles.get(s, 0) / n for s in STYLES},
    }


def print_table(by_cond):
    print(f"{'condition':<14} {'n':>6} {'words':>6}" +
          "".join(f"{s:>13}" for s in STYLES))
    print("-" * 84)
    for cond, s in sorted(by_cond.items()):
        row = "".join(f"{100 * s['styles'][k]:>12.0f}%" for k in STYLES)
        print(f"{cond:<14} {s['n']:>6} {s['mean_words']:>6.2f}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    # A campaign is many timestamped run dirs (one per --resume chunk), so
    # scoping by --runs alone cannot select it. Without this the default sweeps
    # every campaign ever run and compares arms across *different instances* —
    # desc-v5's thousands of historical queries against a new arm's few
    # hundred, unpaired, which is not a comparison at all.
    ap.add_argument("--since", default=None, metavar="RUN_ID",
                    help="only runs whose id sorts >= this (e.g. 20260804-0100)")
    ap.add_argument("--a", default=None, help="treatment arm, for the two-arm delta")
    ap.add_argument("--b", default=None, help="control arm")
    args = ap.parse_args()

    rows, _, _ = harvest(args.runs)
    if args.since:
        rows = [r for r in rows if str(r.get("run_id", "")) >= args.since]
        if not rows:
            print(f"no searches in runs at or after {args.since}")
            return 1
    queries = defaultdict(list)
    for r in rows:
        # Ranked semgrep only: `rg` has no ranked mode, so including it would
        # compare a query language against a regex dialect.
        if r.get("tool") == "rg" or r.get("mode") != "ranked":
            continue
        if r.get("pattern"):
            queries[r["condition"]].append(r["pattern"])

    by_cond = {c: s for c, v in queries.items() if (s := summarize(v))}
    if not by_cond:
        print("no ranked semgrep searches found under", args.runs)
        return 1
    print_table(by_cond)

    if args.a and args.b:
        missing = [c for c in (args.a, args.b) if c not in by_cond]
        if missing:
            print(f"\nno ranked searches for: {', '.join(missing)}")
            return 1
        a, b = by_cond[args.a], by_cond[args.b]
        print(f"\n{args.a} − {args.b}  (n={a['n']} vs {b['n']})")
        print(f"  words/query   {a['mean_words'] - b['mean_words']:+.2f}  "
              f"({a['mean_words']:.2f} vs {b['mean_words']:.2f})")
        for s in STYLES:
            d = 100 * (a["styles"][s] - b["styles"][s])
            print(f"  {s:<13} {d:+.0f}pp  "
                  f"({100 * a['styles'][s]:.0f}% vs {100 * b['styles'][s]:.0f}%)")
        # Deliberately no p-value. These rows cluster within instance and within
        # session, so an unpaired test over them would overstate its confidence
        # by pretending N searches are N independent draws. §19.3 asks whether
        # the description took; a direction with a visible n answers that, and a
        # significance claim would need the clustered treatment ab_analyze.py
        # gives the accuracy endpoints.
        print("\n(direction and magnitude only — these rows cluster within "
              "session, so no test is reported here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
