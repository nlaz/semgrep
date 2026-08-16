#!/usr/bin/env python3
"""Train the §35.2 learned checklist from a guessplay --dump-hits file.

    uv run --with scikit-learn python3 eval/locbench/checklist_train.py \
        --dump eval/data/locbench/guessplay/checklist-dump.jsonl

Pointwise logistic regression over candidate-local features, mirrored exactly
by `search/checklist.rs` — the feature definitions here and there must not
drift, so both cite this list:

    fine_n        fine cosine, min-max normalized within the query's hit list
    fine_missing  1.0 when the fine rerank scored nothing for this hit
    coarse_n      fused score, min-max normalized within the hit list
    bm25_recip    1/bm25_rank, 0.0 when the lexical channel didn't rank it
    bm25_missing  1.0 when it didn't
    phrases_pop   popcount of the retriever bitmask, as f32
    decl_share    structural boost's declaration share (§24.1)
    path_share    structural boost's path share (§35.1)
    chunk_frac    chunk_lines / 32.0 (window-relative height)

Query-global features (query length, mode) are deliberately absent: under a
linear model a per-query constant cannot change within-query order, and
within-query order is the only thing the checklist decides.

Splits are grouped by INSTANCE, never by query — queries within an instance
share gold, and a query-level split leaks (§35.2). The report scores each
held-out fold three ways per query: original engine order, fine-score-only
order, and model order; the §35.2 gate is model > fine-only on held-out MRR.

Emits the Rust const block for `search/checklist.rs` on stdout.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

FEATURES = [
    "fine_n", "fine_missing", "coarse_n", "bm25_recip", "bm25_missing",
    "phrases_pop", "decl_share", "path_share", "chunk_frac",
]


def minmax(vals):
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def featurize(hits):
    """Feature rows for one query's hit list, in hit order."""
    fines = [h["features"].get("fine") for h in hits]
    coarses = [h["features"]["coarse"] for h in hits]
    fine_n = minmax([f if f is not None else 0.0 for f in fines])
    coarse_n = minmax(coarses)
    rows = []
    for i, h in enumerate(hits):
        f = h["features"]
        br = f.get("bm25_rank")
        rows.append([
            fine_n[i] if fines[i] is not None else 0.0,
            0.0 if fines[i] is not None else 1.0,
            coarse_n[i],
            1.0 / br if br else 0.0,
            0.0 if br else 1.0,
            float(bin(f.get("phrases", 1)).count("1")),
            f.get("decl_share", 0.0),
            f.get("path_share", 0.0),
            f.get("chunk_lines", 32) / 32.0,
        ])
    return rows


def order_scores(rows, key):
    """Per-query gold rank under an ordering, as (recall@5, 1/rank or 0)."""
    ranked = sorted(rows, key=key, reverse=True)
    for i, r in enumerate(ranked, 1):
        if r["label"]:
            return (1.0 if i <= 5 else 0.0, 1.0 / i if i <= 10 else 0.0)
    return (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--target", default="label_func_ovl",
                    choices=["label_file", "label_func", "label_func_ovl"])
    ap.add_argument("--modes", default="semantic",
                    help="comma-separated; train on the shipped mode by default")
    ap.add_argument("--conditions", default="",
                    help="comma-separated filter (e.g. desc-v5); empty = all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c", type=float, default=1.0, help="LR inverse regularization")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    modes = set(args.modes.split(","))
    conds = set(args.conditions.split(",")) if args.conditions else None

    queries = []  # one entry per (query invocation): features, labels, instance
    n_rows = n_skipped = 0
    for line in args.dump.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_rows += 1
        if row["mode"] not in modes or (conds and row["condition"] not in conds):
            continue
        hits = [h for h in row.get("hits", []) if h.get("features")]
        if len(hits) < 2:
            n_skipped += 1
            continue
        feats = featurize(hits)
        labels = [bool(h.get(args.target)) for h in hits]
        queries.append({
            "instance": row["instance_id"],
            "rows": [{"x": x, "label": l, "pos": h["pos"],
                      "fine": (h["features"].get("fine") or -2.0)}
                     for x, l, h in zip(feats, labels, hits)],
        })
    print(f"# dump rows {n_rows}, usable queries {len(queries)}, "
          f"skipped(<2 hits or no features) {n_skipped}")

    with_gold = [q for q in queries if any(r["label"] for r in q["rows"])]
    print(f"# queries with a labeled gold hit: {len(with_gold)} "
          f"({100 * len(with_gold) / max(1, len(queries)):.0f}%)")
    if len(with_gold) < 50:
        raise SystemExit("too few labeled queries to train honestly")

    X = np.array([r["x"] for q in with_gold for r in q["rows"]])
    y = np.array([r["label"] for q in with_gold for r in q["rows"]], dtype=int)
    groups = np.array([q["instance"] for q in with_gold for _ in q["rows"]])
    qidx = np.array([i for i, q in enumerate(with_gold) for _ in q["rows"]])

    gkf = GroupKFold(n_splits=args.folds)
    agg = defaultdict(list)
    for tr, te in gkf.split(X, y, groups):
        model = LogisticRegression(C=args.c, max_iter=1000).fit(X[tr], y[tr])
        te_q = sorted(set(qidx[te]))
        scores = X @ model.coef_[0] + model.intercept_[0]
        for qi in te_q:
            rows = with_gold[qi]["rows"]
            idx = np.where(qidx == qi)[0]
            for r, s in zip(rows, scores[idx]):
                r["model"] = float(s)
            for name, key in [("engine", lambda r: -r["pos"]),
                              ("fine", lambda r: r["fine"]),
                              ("model", lambda r: r["model"])]:
                rec, mrr = order_scores(rows, key)
                agg[f"{name}_rec5"].append(rec)
                agg[f"{name}_mrr"].append(mrr)

    n = len(agg["model_mrr"])
    print(f"\n# held-out ({args.folds}-fold, grouped by instance, {n} queries), "
          f"target={args.target}")
    for name in ["engine", "fine", "model"]:
        rec = sum(agg[f"{name}_rec5"]) / n
        mrr = sum(agg[f"{name}_mrr"]) / n
        print(f"#   {name:7s} recall@5 {rec:.4f}   mrr@10 {mrr:.4f}")
    lift = sum(agg["model_mrr"]) / n - sum(agg["fine_mrr"]) / n
    print(f"# model - fine mrr lift: {lift:+.4f}  <- the §35.2 gate-1 number")

    final = LogisticRegression(C=args.c, max_iter=1000).fit(X, y)
    print("\n// ---- paste into crates/semgrep-core/src/search/checklist.rs ----")
    print(f"// trained {Path(args.dump).name}, target={args.target}, "
          f"C={args.c}, {len(with_gold)} queries")
    print("pub(crate) const WEIGHTS: [f32; %d] = [" % len(FEATURES))
    for name, w in zip(FEATURES, final.coef_[0]):
        print(f"    {w:.6}f32, // {name}")
    print("];")
    print(f"pub(crate) const BIAS: f32 = {final.intercept_[0]:.6}f32;")


if __name__ == "__main__":
    main()
