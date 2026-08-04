#!/usr/bin/env python3
"""hit@5 cut by query style — the measurement behind RESEARCH.md §19.2b.

The question is what a *static* embedding model does with a paraphrase. `ese`
is one vector per token, SIF-pooled, word order discarded, and `sif.rs` weights
tokens `a/(a + p(w))` with `a = 1e-3` — a word in 1% of the corpus scores 0.09
against a rare one's 0.99. So a description reduces, at the engine, to its rare
tokens, and this script asks what that costs when those tokens miss.

Checked in because §19.2b's numbers decide a campaign's design, and a finding
that only exists in someone's shell history is not reproducible.

    python3 eval/locbench/stylecut.py              # §19.2b's two tables
    python3 eval/locbench/stylecut.py --arm all    # include t1/t2 translations

Three controls, each of which changes the answer:

1. **Arm.** `t1`/`t2` rows are harness translations of grep patterns and are
   identifier-shaped by construction, from agents who already had a symbol.
   Only `ranked-own` is the agent's own ranked phrasing, and it is the default.
2. **Scope.** File-scoped rows predate the §16.11 fix and are all zero, so
   including them would score the bug rather than the query.
3. **Gold vocabulary overlap.** §17.4's predictor. A query carrying the gold's
   own tokens was written by someone who already knew something, so style and
   knowledge have to be separated before either can be read.

What this is not: randomised. Agents chose how to phrase, so overlap is a
control and not an assignment. n is small in the stratum that carries the
result. The direction is far better established than the magnitude.
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"

FUNC = {"where", "is", "the", "a", "an", "how", "what", "does", "do", "in",
        "to", "for", "of", "and", "are", "when", "which", "that", "this", "it",
        "on", "with", "by", "from", "be", "was", "were", "or", "if", "then",
        "we", "you", "i", "its", "their", "there", "here", "why", "who", "can",
        "should", "would", "get", "set"}
SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")
CAMEL = re.compile(r"^[a-z]+([A-Z][a-z0-9]*)+$|^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)+$")
DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
WORD = re.compile(r"[a-z0-9]+")
STYLES = ["identifiers", "plain words", "mixed", "paraphrase"]


def code_token(tok):
    tok = tok.strip("(),:\"'`")
    return bool(SNAKE.match(tok) or CAMEL.match(tok) or DOTTED.match(tok))


def classify(query):
    toks = query.split()
    if not toks:
        return None
    code = sum(1 for t in toks if code_token(t))
    func = sum(1 for t in toks if t.lower().strip(".,?") in FUNC)
    if code and not func: return "identifiers"
    if func and not code: return "paraphrase"
    if code and func:     return "mixed"
    return "plain words"


def is_description(query):
    """The robust binary: English function words present.

    Preferred over `classify` for the headline because it does not depend on
    recognising code shape — `cpp_appendColumnToParquet` matches neither the
    snake_case nor the camelCase pattern and falls into "plain words", making
    that class a mixture of prose and unrecognised identifiers. Function words
    have no such middle.
    """
    return any(t.lower().strip(".,?") in FUNC for t in query.split())


def subtokens(s):
    """Split code-style, so get_config and getConfig both -> {get, config}."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return {w for w in WORD.findall(s.lower()) if len(w) > 2}


def scope_shape(s):
    if not s or s in (".", ""): return "root"
    return "file" if os.path.splitext(s)[1] else "dir"


def boot(vals, n=4000, seed=7):
    """Bootstrap CI on a mean. Seeded: a published CI must not move per run."""
    rnd = random.Random(seed)
    m = sum(vals) / len(vals)
    ms = []
    for _ in range(n):
        s = [vals[rnd.randrange(len(vals))] for _ in range(len(vals))]
        ms.append(sum(s) / len(s))
    ms.sort()
    return m, ms[int(.025 * n)], ms[int(.975 * n)]


def load(args):
    gold = {}
    for line in open(args.dataset):
        d = json.loads(line)
        gold[d["instance_id"]] = d.get("edit_functions") or []
    rows = [json.loads(l) for l in open(args.guessplay)]
    frame = [r for r in rows
             if r["config"] == "default" and r["scope_policy"] == "orig"
             and r["mode"] in ("semantic", "bm25", "hybrid")
             and scope_shape(r.get("scope")) != "file"
             and not r["dead"] and r.get("query")]
    by = defaultdict(dict)
    for r in frame:
        by[(r["gid"], r["arm"])][r["mode"]] = r
    paired = [v for v in by.values() if {"semantic", "bm25", "hybrid"} <= set(v)]
    if args.arm != "all":
        paired = [v for v in paired if v["semantic"]["arm"] == args.arm]
    return paired, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guessplay", type=Path, default=DATA / "guessplay.jsonl")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--arm", default="ranked-own",
                    help="ranked-own (the agent's own queries) | t1 | all")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    paired, gold = load(args)
    if not paired:
        print("no paired rows — check --arm")
        return 1

    def hit(r): return bool(r["rank"] and r["rank"] <= args.k)

    def overlap(v):
        q, iid = v["semantic"]["query"], v["semantic"]["instance_id"]
        g = set()
        for f in gold.get(iid, []):
            g |= subtokens(f)
        return None if not g else bool(subtokens(q) & g)

    print(f"arm={args.arm}  paired queries={len(paired)}  (hit@{args.k})\n")

    print("four-way style, all three rankers")
    print(f"{'style':<13}{'n':>5}{'words':>7}{'semantic':>10}{'bm25':>8}{'hybrid':>8}")
    g = defaultdict(list)
    for v in paired:
        g[classify(v["semantic"]["query"])].append(v)
    for s in STYLES:
        v = g.get(s)
        if not v: continue
        print(f"{s:<13}{len(v):>5}"
              f"{sum(len(x['semantic']['query'].split()) for x in v)/len(v):>7.1f}"
              f"{sum(hit(x['semantic']) for x in v)/len(v):>10.3f}"
              f"{sum(hit(x['bm25']) for x in v)/len(v):>8.3f}"
              f"{sum(hit(x['hybrid']) for x in v)/len(v):>8.3f}")

    print("\ncollapsed: does the query contain English function words?")
    print("(the headline — no dependence on recognising code shape)")
    print(f"{'':<24}{'name-like':>26}{'description':>26}")
    for ov, lbl in ((None, "all"), (True, "shares gold vocab"),
                    (False, "shares NO gold vocab")):
        cells = []
        for desc in (False, True):
            sub = [v for v in paired
                   if is_description(v["semantic"]["query"]) == desc
                   and (ov is None or overlap(v) is ov)]
            if len(sub) < 10:
                cells.append(f"{'n=' + str(len(sub)) + ' (too few)':>26}")
                continue
            m, lo, hi = boot([hit(x["hybrid"]) for x in sub])
            cells.append(f"{m:.3f} n={len(sub):<4} [{lo:.2f},{hi:.2f}]".rjust(26))
        print(f"{lbl:<24}" + "".join(cells))

    print("\nselection effect — how often each style reaches gold vocabulary:")
    for s in STYLES:
        v = [x for x in g.get(s, []) if overlap(x) is not None]
        if len(v) < 10: continue
        k = sum(1 for x in v if overlap(x))
        print(f"  {s:<13}{k:>4}/{len(v):<4} = {100*k/len(v):>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
