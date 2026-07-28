#!/usr/bin/env python3
"""Scoring for Loc-Bench localization answers.

Gold: edit_functions entries like "rel/path.py:Class.method" (file-only
entries have no ':'). Predictions: the agent's {"files": [...],
"functions": ["path:qual", ...]} lists, ordered most-likely first.

Function matching is reported at two strictnesses (both stored, so
over-credit from sloppy agent output is auditable rather than silent):
  strict   — exact qualname equality
  tolerant — + dot-suffix either way (agent says `method` for gold
             `Class.method` or vice versa), + leaf-name equality
"""

import json


def parse_gold(edit_functions):
    """-> (gold_files: [path], gold_funcs: [(path, qualname)])"""
    files, funcs = [], []
    for entry in edit_functions:
        if ":" in entry:
            path, qual = entry.split(":", 1)
            funcs.append((norm_path(path), qual))
            path = norm_path(path)
        else:
            path = norm_path(entry)
        if path not in files:
            files.append(path)
    return files, funcs


def norm_path(p):
    p = p.strip().replace("\\", "/").lstrip("./")
    # Agents sometimes answer with absolute worktree paths; keep the
    # repo-relative tail by dropping anything before a known marker-less
    # absolute prefix. Callers with the worktree path should pre-strip;
    # this is the generic fallback.
    return p.lstrip("/") if p.startswith("/") else p


def file_match(pred, gold_files):
    """Exact relpath match, else unique-suffix match (agent dropped a
    leading dir). Returns the matched gold path or None."""
    if pred in gold_files:
        return pred
    suffix_hits = [g for g in gold_files if g.endswith("/" + pred) or pred.endswith("/" + g)]
    return suffix_hits[0] if len(suffix_hits) == 1 else None


def func_match(pred_qual, gold_qual, tolerant):
    if pred_qual == gold_qual:
        return True
    if not tolerant:
        return False
    if pred_qual.endswith("." + gold_qual) or gold_qual.endswith("." + pred_qual):
        return True
    return pred_qual.split(".")[-1] == gold_qual.split(".")[-1]


def parse_pred_functions(functions, worktree_prefix=None):
    """-> ([(path, qual)], n_malformed). Entries without ':' are malformed
    (file-only noise in the functions list) and are counted, not matched."""
    out, malformed = [], 0
    for entry in functions:
        entry = str(entry)
        if worktree_prefix and entry.startswith(worktree_prefix):
            entry = entry[len(worktree_prefix):].lstrip("/")
        if ":" not in entry:
            malformed += 1
            continue
        path, qual = entry.split(":", 1)
        out.append((norm_path(path), qual.strip()))
    return out, malformed


def score_instance(answer_files, answer_functions, gold_files, gold_funcs, worktree_prefix=None):
    """-> metrics dict. Acc@k = ALL gold files/functions covered by the
    top-k predictions (LocAgent-comparable 'complete localization')."""
    preds = []
    for p in answer_files:
        p = str(p)
        if worktree_prefix and p.startswith(worktree_prefix):
            p = p[len(worktree_prefix):].lstrip("/")
        preds.append(norm_path(p))
    pred_funcs, malformed = parse_pred_functions(answer_functions, worktree_prefix)

    m = {"malformed_preds": malformed}

    # File-level: which gold files are hit within the top-k predictions.
    def gold_files_hit(k):
        hit = set()
        for p in preds[:k]:
            g = file_match(p, gold_files)
            if g:
                hit.add(g)
        return hit

    for k in (1, 3, 5):
        m[f"file_acc@{k}"] = int(bool(gold_files) and gold_files_hit(k) == set(gold_files))
    m["file_recall@5"] = (
        len(gold_files_hit(5)) / len(gold_files) if gold_files else None
    )

    # Function-level, strict and tolerant.
    def gold_funcs_hit(k, tolerant):
        hit = set()
        for pf_path, pf_qual in pred_funcs[:k]:
            for i, (gf_path, gf_qual) in enumerate(gold_funcs):
                if i in hit:
                    continue
                if file_match(pf_path, [gf_path]) and func_match(pf_qual, gf_qual, tolerant):
                    hit.add(i)
                    break
        return hit

    for label, tolerant in (("strict", False), ("tol", True)):
        for k in (5, 10):
            m[f"func_acc@{k}_{label}"] = int(
                bool(gold_funcs) and gold_funcs_hit(k, tolerant) == set(range(len(gold_funcs)))
            )
        m[f"func_recall@10_{label}"] = (
            len(gold_funcs_hit(10, tolerant)) / len(gold_funcs) if gold_funcs else None
        )
    return m


def first_gold_hit_seq(shim_log_path, stdout_dir, gold_files):
    """Position (1-based) of the first *real* search whose output mentions a
    gold file path — searches-to-first-useful-hit. Blocked invocations
    (grep/git steered away) don't count as searches. None = no search ever
    surfaced a gold file. Matches the path's dir/base tail to tolerate
    absolute/relative prefixes in tool output."""
    import pathlib

    log = pathlib.Path(shim_log_path)
    if not log.exists():
        return None
    tails = [g.split("/")[-1] if "/" not in g else "/".join(g.split("/")[-2:]) for g in gold_files]
    rows = []
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    pos = 0
    for row in sorted(rows, key=lambda r: r["seq"]):
        if row.get("blocked"):
            continue
        pos += 1
        f = pathlib.Path(stdout_dir) / row["stdout_file"]
        if not f.exists():
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if any(t in text for t in tails):
            return pos
    return None
