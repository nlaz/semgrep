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

import re
import posixpath
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


# Flags that consume the next argv token, so a scope is not confused with a
# flag's value. Superset across rg and semgrep; a flag missing here can only
# cost a scope, never invent one.
_VALUED_FLAGS = {"-k", "--top", "-C", "--context", "-A", "-B", "-M", "--max-columns",
                 "-g", "--glob", "--include", "-m", "--max-count", "-t", "--type",
                 "--mode", "-e", "--regexp", "-f", "--file", "--max-depth"}
_HIT_LINE = re.compile(r"(?m)^([^\s:][^:\n]*):\d+:")
# The single-file form, where the tool omits the path because the caller
# already named it: `264:\ttext`. Without this the scorer goes blind on
# exactly the scope agents use most — see the docstring below.
_HIT_LINE_NOPATH = re.compile(r"(?m)^\d+:\t")


def _scopes(argv):
    """Positional path arguments — what the tool printed its output relative to."""
    pos, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-"):
            i += 2 if a in _VALUED_FLAGS else 1
            continue
        pos.append(a)
        i += 1
    return pos[1:]  # pos[0] is the query/pattern


def _rel(path, cwd):
    if cwd and path.startswith(cwd):
        path = path[len(cwd):]
    return path.lstrip("/").lstrip("./")


def first_gold_hit_seq(shim_log_path, stdout_dir, gold_files):
    """Position (1-based) of the first *real* search whose output surfaced a
    gold file — searches-to-first-useful-hit. Blocked invocations (grep/git
    steered away) don't count as searches. None = no search ever surfaced one.

    Two matchers, because one tool's output shape defeated the other.

    The tail match (`dir/base` anywhere in the text) tolerates the absolute
    prefixes rg prints. It cannot see semgrep's output at all when the agent
    scopes below the repo root: **semgrep prints paths relative to the scope it
    was given, and rg prints them as passed.** `semgrep q msal/` yields
    `application.py:162:…` where `rg q msal/` yields `msal/application.py:162:…`,
    so a two-component tail matches ripgrep and misses semgrep — a systematic
    undercount of one arm only. On the §19.7 campaign it hid a gold hit in 13 of
    204 desc-v8 rows against 5 of 204 rg rows, including one where every one of
    the agent's four searches returned the gold file and the metric read `None`.

    So each result line's path is also rejoined to the invocation's own scope
    and compared to gold outright. Both matchers are ORed: either is sufficient
    evidence the search surfaced the file.

    A third form exists since the tool started following grep's single-file
    rule: given one named file it prints `264\ttext` with no path at all,
    because the path is on every line and the caller just typed it. Neither
    matcher above can see that, so the scope itself is checked against gold —
    a hit in a path-less listing is in the named file by construction.
    """
    import pathlib

    log = pathlib.Path(shim_log_path)
    if not log.exists():
        return None
    gold = set(gold_files)
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
        f = pathlib.Path(stdout_dir) / (row.get("stdout_file") or "")
        if not f.exists():
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if any(t in text for t in tails):
            return pos
        cwd = row.get("cwd") or ""
        scopes = [_rel(s, cwd) for s in _scopes(row.get("argv") or [])]
        # Path-less output means the caller named exactly one file, so the hit
        # is in *that* file by construction — there is nothing else it could be.
        if _HIT_LINE_NOPATH.search(text) and any(s in gold for s in scopes):
            return pos
        for m in _HIT_LINE.finditer(text):
            printed = _rel(m.group(1), cwd)
            if printed in gold:
                return pos
            for s in scopes:
                if not s:
                    continue
                # A file scope prints as its own basename; a dir scope prefixes.
                if s in gold and printed == s.split("/")[-1]:
                    return pos
                if posixpath.normpath(posixpath.join(s, printed)) in gold:
                    return pos
    return None
