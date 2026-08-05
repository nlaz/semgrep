#!/usr/bin/env python3
"""Translate-and-compare: the §16 product-hypothesis test.

For every guess-group in the corpus (one exact/rg invocation's ladder; a
bare guess is a 1-rung ladder), replay three arms against the instance's
gold files:

  exact       the agent's actual pattern, verbatim, through `semgrep -e`
  exact-norm  dead `\\|` ladders re-run with `|` — what the agent MEANT
  t1 / t2     the ranked translations, under --mode bm25|semantic|hybrid

plus the agents' real ranked queries (kind=guess_ranked) re-scored under the
same build as the reference distribution. Scope policy: `orig` uses the
invocation's own first scope (93% of calls are scoped); `root` is the
sensitivity cut. Config `champion` rebuilds the worktree index with the §14
flags before its pass — measured because §15.9-B says SIF is mis-tuned for
token-poor queries, and guesses are token-poor.

    python3 eval/locbench/guessplay.py --limit-instances 3      # smoke
    python3 eval/locbench/guessplay.py                          # full

Output: one row per (group, arm, scope) appended to
eval/data/locbench/guessplay.jsonl (checkpointing: existing keys skipped);
convert to a results-board cell file with --emit-results.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ladder  # noqa: E402
import run as locbench  # noqa: E402
import scoring  # noqa: E402
import symbols as symmod  # noqa: E402
from replay import gold_files, rank_of_gold  # noqa: E402

_SYMS = {}

DATA = HERE.parent / "data" / "locbench"
QUERIES = HERE.parent / "queries" / "guesses-v0.jsonl"
# Rendering configs (§21). Each carries BOTH halves of the dose.
#
# The index half is the lever for a *directory* scope: the warm path renders
# with the index's stored `meta.embed_preproc` and ignores the flag entirely
# (search/indexed.rs:211). The search half is the lever for a *file* scope:
# `cache::discover` bails on a non-directory root (cache/mod.rs:74-76), so a
# file-scoped search finds no index at all and the cold path reads the flag
# (search/stream.rs:77,126). Wiring only one of the two silently leaves a
# share of the corpus untreated and reports a diluted null.
#
# `--sif` is index-only (cli.rs, Cmd::Index) and `stream.rs` has no SIF pass,
# so `champion` is *partially treatable* by construction: file-scoped rows get
# split without sif. Recorded here rather than hidden, and reported as such.
CONFIGS = {
    "default":    {"index": [],                                   "search": []},
    "split":      {"index": ["--embed-preproc", "split"],
                   "search": ["--embed-preproc", "split"]},
    "prune-kw":   {"index": ["--embed-preproc", "prune-kw"],
                   "search": ["--embed-preproc", "prune-kw"]},
    "prune-decl": {"index": ["--embed-preproc", "prune-decl"],
                   "search": ["--embed-preproc", "prune-decl"]},
    # §22's 2x2. Both render DOCUMENTS identically; they differ only in whether
    # the keyword table also fires on the query, which is the axis under test.
    "prune-kw-pos":    {"index": ["--embed-preproc", "prune-kw-pos"],
                        "search": ["--embed-preproc", "prune-kw-pos"]},
    "prune-kw-pos-q0": {"index": ["--embed-preproc", "prune-kw-pos-q0"],
                        "search": ["--embed-preproc", "prune-kw-pos-q0"]},
    "champion":   {"index": ["--embed-preproc", "split", "--sif"],
                   "search": ["--embed-preproc", "split"],
                   "partial": "sif is index-only; file scopes get split alone"},
}
MODES = ("bm25", "semantic", "hybrid")

# Which binary produced a row. Without it, resuming into a file written by an
# older binary compares two engines and calls it a config delta: the existing
# guessplay.jsonl predates the §16.11 file-scope fix, and 208 of its 624
# ranked rows are hardcoded zeros as a result.
BIN_SHA = hashlib.sha256(locbench.SEMGREP.read_bytes()).hexdigest()[:16]


def gid(row):
    return f"{row['run_id']}/{row['instance_id']}/{row['condition']}/{row['seq']}"


def run_semgrep(tree, scope_rel, query, k, is_exact, mode, cache_dir,
                search_flags=()):
    path = tree if scope_rel in (None, ".") else tree / scope_rel
    cmd = [str(locbench.SEMGREP), "--json", "-k", str(k)]
    if is_exact:
        cmd.append("-e")
    else:
        cmd += ["--mode", mode, *search_flags]
    cmd += [query, str(path)]
    env = dict(os.environ)
    env["SEMGREP_CACHE_DIR"] = str(cache_dir)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode == 2:
        return None, "error"  # e.g. the agent's scope path doesn't exist
    hits = []
    for line in p.stdout.splitlines():
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return hits, None


def _abs_hits(hits, scope_rel, tree):
    """Hits with repo-relative paths, the shape gold entries use.

    Same composition `score()` does, factored out so the file-scope rule that
    §21.2 got wrong lives in exactly one place: a FILE scope is not a directory
    and must not be prefixed.
    """
    if hits is None:
        return []
    if scope_rel in (None, "."):
        return hits
    is_file = (tree / scope_rel).is_file() if tree is not None else bool(
        PurePosixPath(scope_rel).suffix)
    if is_file:
        return [{**h, "path": scope_rel} for h in hits]
    prefix = scope_rel.rstrip("/") + "/"
    return [{**h, "path": prefix + h.get("path", "")} for h in hits]


def _symbols_for(tree, rel):
    """`symbols.extract` for one repo-relative file, memoized.

    Without the cache every hit re-parses the whole file - a 60-symbol module
    parsed once per hit per query per config.
    """
    key = (str(tree), rel)
    if key not in _SYMS:
        f = tree / rel
        try:
            _SYMS[key] = symmod.extract(f) if f.is_file() else []
        except (OSError, ValueError):
            _SYMS[key] = []
    return _SYMS[key]


def _enclosing(syms, line):
    """Innermost symbol whose declaration span contains `line`, else None.

    `sig_line..end_line`, not `start_line..end_line`: start_line includes the
    leading doc/decorator block, so a hit on a docstring above one function
    would be credited to it. Spans nest (a class encloses its methods), so the
    innermost - smallest span - is the answer.
    """
    best = None
    for sym in syms:
        if sym["sig_line"] <= line <= sym["end_line"]:
            span = sym["end_line"] - sym["sig_line"]
            if best is None or span < best[0]:
                best = (span, sym["name"])
    return best[1] if best else None


def rank_of_gold_func(hits, gold_funcs, tree, scope_rel):
    """1-based rank of the first hit landing in a gold FUNCTION, else None.

    Why this exists: under a file scope every hit carries that one file's path,
    so `rank_of_gold` is 1-or-absent however the engine orders chunks - the
    rank histogram over 2,928 file-scoped rows was exactly {1: ..., None: ...}
    and nothing else. That is a metric artifact, not an engine property (§22.1).
    Within-file chunk order still decides which FUNCTIONS the agent sees, and
    `func_acc@10_tol` is the endpoint this project actually reports.

    Tolerant matching only. `symbols.extract` yields bare leaf names and 704 of
    1,149 gold quals are dotted `Class.method`, so the leaf-name clause of
    `scoring.func_match` is the bridge and `_strict` is not computable here.

    A hit whose line falls in no extracted symbol is skipped, not counted as a
    miss: the extractor under-counts by design (multi-line signatures), and
    conflating "no function here" with "wrong function" would bias every arm
    identically but silently.
    """
    if not gold_funcs:
        return None
    for i, h in enumerate(hits, 1):
        rel = h.get("path", "")
        if not rel:
            continue
        # `line` is the best-matching line WITHIN the chunk. Containment on it,
        # not span overlap: the 32-line window straddles several small Python
        # functions, and overlap credit would blunt the very ordering this is
        # built to measure.
        line = h.get("line") or h.get("start_line")
        if not line:
            continue
        name = _enclosing(_symbols_for(tree, rel), line)
        if name is None:
            continue
        for gpath, gqual in gold_funcs:
            if scoring.file_match(rel, [gpath]) and scoring.func_match(name, gqual, True):
                return i
    return None


def scope_of(row, policy):
    if policy == "root" or not row["scopes_rel"]:
        return "."
    return row["scopes_rel"][0]


def score(hits, golds, scope_rel, tree=None):
    """Hits are scope-relative when scoped; golds are repo-relative.

    A FILE scope is not a directory and must not be prefixed. `corpus::walk`
    records a file-scoped root's own basename as the relative path, so the hit
    comes back as `trainer.py` for a scope of `pkg/trainer.py`; prefixing gives
    `pkg/trainer.py/trainer.py`, which matches no gold and scores every
    file-scoped row as a miss. 46% of desc-v9's ranked searches are file-scoped
    (§21.1), so this silently zeroed nearly half the corpus in every arm — and
    it is a scoring bug, not the §16.11 engine bug it was mistaken for.
    """
    if hits is None:
        return None
    if scope_rel not in (None, "."):
        is_file = (tree / scope_rel).is_file() if tree is not None else bool(
            PurePosixPath(scope_rel).suffix)
        if is_file:
            hits = [{**h, "path": scope_rel} for h in hits]
        else:
            prefix = scope_rel.rstrip("/") + "/"
            hits = [{**h, "path": prefix + h.get("path", "")} for h in hits]
    return rank_of_gold(hits, golds)


def arms_for(row, modes=MODES):
    """(arm_name, query, is_exact, modes) tuples for one corpus row."""
    if row["kind"] == "guess_ranked":
        return [("ranked-own", row["pattern"], False, modes)]
    lad = ladder.parse(row["patterns"])
    out = [("exact", row["pattern"], True, (None,))]
    if lad["engine_semantics_mismatch"]:
        out.append(("exact-norm", row["pattern"].replace("\\|", "|"), True, (None,)))
    t1 = ladder.translate_t1(lad)
    if t1:
        out.append(("t1", t1, False, modes))
        t2 = ladder.translate_t2(lad)
        if t2 and t2 != t1.lower() and "hybrid" in modes:
            out.append(("t2", t2, False, ("hybrid",)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=QUERIES)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "guessplay.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--configs", default="default,champion")
    ap.add_argument("--scopes", default="orig,root")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="ranked modes to score; the §21 gate needs only "
                         "semantic (shipped) and bm25 (the tripwire)")
    ap.add_argument("--limit-instances", type=int, default=0)
    ap.add_argument("--instances", default="")
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--emit-results", type=Path, default=None,
                    help="aggregate an existing --out into a results-board "
                         "cell file and exit")
    ap.add_argument("--emit-config", default="default",
                    help="which index config --emit-results should aggregate")
    ap.add_argument("--compare-metrics", default="rank",
                    help="rank (gold file) and/or rank_func (gold function)")
    ap.add_argument("--compare", default="",
                    help="BASE,CAND[,CAND...] - print the §21 gate report from "
                         "an existing --out and exit")
    args = ap.parse_args()

    if args.compare:
        for fld in args.compare_metrics.split(","):
            compare(args.out, args.compare.split(","), field=fld)
        return
    if args.emit_results:
        emit_results(args.out, args.emit_results, args.emit_config)
        return

    ds = {}
    for line in args.dataset.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["instance_id"]] = r

    rows = [json.loads(l) for l in args.corpus.read_text().splitlines()]
    rows = [r for r in rows if r["instance_id"] in ds]
    by_instance = defaultdict(list)
    for r in rows:
        by_instance[r["instance_id"]].append(r)
    instances = sorted(by_instance)
    if args.instances:
        want = set(args.instances.split(","))
        instances = [i for i in instances if i in want]
    if args.limit_instances:
        instances = instances[: args.limit_instances]

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                e = json.loads(line)
                done.add((e["gid"], e["arm"], e["mode"], e["config"],
                          e["scope_policy"], e.get("bin_sha256")))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"{len(instances)} instances, {sum(len(by_instance[i]) for i in instances)} "
          f"guess rows; {len(done)} arm-rows already done")

    configs = args.configs.split(",")
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown --configs {unknown}; have {sorted(CONFIGS)}")
    want_modes = tuple(args.modes.split(","))
    scope_policies = args.scopes.split(",")
    tmp = tempfile.TemporaryDirectory(prefix="semgrep-guessplay-cache-")
    cache_dir = Path(tmp.name)

    out_f = open(args.out, "a")
    n_run = 0
    for n_i, inst_id in enumerate(instances, 1):
        inst = ds[inst_id]
        golds = gold_files(inst)
        _, gold_funcs = scoring.parse_gold(inst.get("edit_functions") or [])
        if not golds:
            continue
        try:
            tree, _ = locbench.ensure_worktree(inst["repo"], inst["base_commit"])
            meta_path = (locbench.DATA / "repos" / "meta" /
                         f"{locbench.repo_key(inst['repo'], inst['base_commit'])}.json")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {inst_id}: {type(e).__name__}: {e}")
            continue
        for config in configs:
            # One index per (worktree, config); exact arms don't read it but
            # ranked arms must all see the same build within a config.
            flags = CONFIGS[config]["index"]
            subprocess.run([str(locbench.SEMGREP), "index", str(tree), *flags],
                           check=True, capture_output=True, timeout=600)
            # Readback: a failed or ignored build used to degrade silently into
            # "measured the previous config", which is indistinguishable from a
            # null. Assert the index is in the space the arm asked for.
            got = json.loads((tree / ".semgrep" / "meta.json").read_text())
            want_pp = "none"
            if "--embed-preproc" in flags:
                want_pp = flags[flags.index("--embed-preproc") + 1]
            assert got.get("embed_preproc", "none") == want_pp and \
                   got.get("sif", False) == ("--sif" in flags), (
                       f"{config}: index built as "
                       f"{got.get('embed_preproc')}/sif={got.get('sif')}, "
                       f"wanted {want_pp}/sif={'--sif' in flags}")
            for row in by_instance[inst_id]:
                for policy in scope_policies:
                    if config != "default" and policy != scope_policies[0]:
                        continue  # variants measured under the primary policy only
                    sc = scope_of(row, policy)
                    for arm, query, is_exact, modes in arms_for(row, want_modes):
                        if is_exact and config != "default":
                            continue  # keyword path ignores index AND flag
                        for mode in modes:
                            key = (gid(row), arm, mode or "-", config, policy,
                                   BIN_SHA)
                            if key in done:
                                continue
                            hits, err = run_semgrep(
                                tree, sc, query, args.k, is_exact, mode, cache_dir,
                                search_flags=CONFIGS[config]["search"])
                            out_f.write(json.dumps({
                                "gid": gid(row), "instance_id": inst_id,
                                "kind": row["kind"], "condition": row["condition"],
                                "arm": arm, "mode": mode or "-", "config": config,
                                "scope_policy": policy, "scope": sc,
                                "n_rungs": ladder.parse(row["patterns"])["n_rungs"],
                                "dead": "\\|" in row["pattern"],
                                "query": query, "rank": score(hits, golds, sc, tree),
                                "rank_func": (
                                    None if is_exact else
                                    rank_of_gold_func(_abs_hits(hits, sc, tree),
                                                      gold_funcs, tree, sc)),
                                "err": err, "bin_sha256": BIN_SHA,
                            }, sort_keys=True) + "\n")
                            out_f.flush()
                            n_run += 1
        if not args.keep_worktrees:
            locbench.remove_worktree(inst["repo"], inst["base_commit"])
        if n_i % 5 == 0:
            print(f"  {n_i}/{len(instances)} instances, {n_run} arm-rows", flush=True)
    out_f.close()
    print(f"done: {n_run} new arm-rows in {args.out}")


def emit_results(raw_path, out_path, config="default"):
    """Aggregate guessplay rows into run_eval cell shape for the Guess board."""
    rows = [json.loads(l) for l in raw_path.read_text().splitlines()]
    cells = defaultdict(list)
    for r in rows:
        if r["scope_policy"] != "orig" or r["config"] != config:
            continue
        mode = r["arm"] if r["arm"].startswith("exact") else f"{r['arm']}-{r['mode']}"
        cells[(mode, r["kind"])].append((r["instance_id"], r["rank"]))
    out = []
    for (mode, kind), pairs in sorted(cells.items()):
        ranks = [rk for _, rk in pairs]
        n = len(ranks)
        rec = lambda k: sum(1 for r in ranks if r and r <= k) / n
        mrr = sum(1 / r for r in ranks if r and r <= 10) / n
        out.append({
            "mode": mode, "kind": kind, "n": n,
            "recall@1": rec(1), "recall@5": rec(5), "recall@10": rec(10),
            "mrr@10": mrr,
            "ranks": ranks,
            "instances": [i for i, _ in pairs],
            "run": {"queries": raw_path.name, "corpus": "(locbench worktrees)",
                    "scope_policy": "orig", "config": config},
        })
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} cells to {out_path}")


def _is_file_scope(scope):
    """A file scope finds no index (cache/mod.rs:74) and, before the §16.11
    fix, found no gold either - 208 of the v0 corpus's 624 ranked rows are
    hardcoded zeros. Reported separately, never pooled."""
    s = scope or "."
    return s not in (".", "") and "." in s.rsplit("/", 1)[-1]


def _cluster_ci(by_instance, n_resamples=4000, seed=1):
    """Percentile CI on mean(cand) - mean(base), resampling INSTANCES.

    Queries are clustered inside instances (median 11, max 59 per instance),
    so a per-query bootstrap understates the width - measured design effect
    1.64x on this corpus, enough to flip the champion from null to
    'significant'. Same conventions as ab_analyze.boot_ci: 4,000, seed 1.
    """
    import random
    ids = list(by_instance)
    if not ids:
        return None, None, None
    flat = lambda sel: [x for i in sel for x in by_instance[i]]
    mean = lambda q, j: (sum(p[j] for p in q) / len(q)) if q else 0.0
    q0 = flat(ids)
    point = mean(q0, 1) - mean(q0, 0)
    rng = random.Random(seed)
    ds = []
    for _ in range(n_resamples):
        q = flat([rng.choice(ids) for _ in ids])
        if q:
            ds.append(mean(q, 1) - mean(q, 0))
    ds.sort()
    return point, ds[int(0.025 * len(ds))], ds[min(int(0.975 * len(ds)), len(ds) - 1)]


def _mcnemar(b, c):
    """Exact two-sided sign test over discordant pairs, as ab_analyze does."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def compare(raw_path, configs, k=5, field="rank"):
    """The §21 phase-1 gate report.

    The gate is psi_offline - the share of INSTANCES whose outcome the arm
    changes - not offline recall. Recall is the quantity §9.7, §10.6 and §14.5
    each established does not transfer; instance-level discordance is what
    McNemar power is actually made of, so an arm with psi_offline = 0 cannot be
    measured at agent level at any price.
    """
    base, cands = configs[0], configs[1:]
    rows = [json.loads(l) for l in raw_path.read_text().splitlines()]
    shas = {r.get("bin_sha256") for r in rows if r.get("config") in configs}
    print(f"corpus: {raw_path.name}")
    print(f"binaries: {sorted(x for x in shas if x)}"
          f"{'   <-- TRIPWIRE: more than one binary produced these rows' if len(shas) > 1 else ''}")

    for mode in ("semantic", "bm25"):
        sel = [r for r in rows if r.get("arm") == "ranked-own"
               and r.get("mode") == mode and r.get("scope_policy") == "orig"]
        by_gid = defaultdict(dict)
        for r in sel:
            by_gid[r["gid"]][r["config"]] = r
        hit = lambda r: bool(r.get(field) and r[field] <= k)

        print(f"\n=== mode={mode}  metric={field}@{k}  (base={base})")
        hdr = (f"{'arm':<17}{'scope':<11}{'n':>5}{'base':>8}{'cand':>8}{'delta':>8}"
               f"{'cluster 95% CI':>20}   instances  b/c  psi   p")
        print(hdr); print("-" * len(hdr))
        for cand in cands:
            pairs = [(v[base], v[cand]) for v in by_gid.values()
                     if base in v and cand in v]
            for label, keep in (("dir/root", lambda d: not _is_file_scope(d.get("scope"))),
                                ("file", lambda d: _is_file_scope(d.get("scope")))):
                p = [(d, c) for d, c in pairs if keep(d)]
                if not p:
                    continue
                byi = defaultdict(list)
                for d, c in p:
                    byi[d["instance_id"]].append((hit(d), hit(c)))
                pt, lo, hi = _cluster_ci(byi)
                bm = sum(1 for d, _ in p if hit(d)) / len(p)
                cm = sum(1 for _, c in p if hit(c)) / len(p)
                # instance-level: did the arm change THIS instance's outcome?
                inst = {}
                for d, c in p:
                    a = inst.setdefault(d["instance_id"], [False, False])
                    a[0] |= hit(d); a[1] |= hit(c)
                b_ = sum(1 for v in inst.values() if v[1] and not v[0])
                c_ = sum(1 for v in inst.values() if v[0] and not v[1])
                psi = (b_ + c_) / len(inst) if inst else 0.0
                ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "-"
                print(f"{cand:<17}{label:<11}{len(p):>5}{bm:>8.3f}{cm:>8.3f}{pt:>+8.3f}"
                      f"{ci:>20}   {len(inst):>9}  {b_}/{c_}  {psi:.3f}  "
                      f"{_mcnemar(b_, c_):.3f}")

        # §21 P3: the dose law predicts the loss shrinks as queries get shorter
        if mode == "semantic":
            print(f"\n  by query length (dir/root only) - the §20.8 dose test")
            print(f"  {'arm':<17}{'1w':>10}{'2w':>10}{'3-4w':>10}{'5+w':>10}")
            for cand in cands:
                pairs = [(v[base], v[cand]) for v in by_gid.values()
                         if base in v and cand in v
                         and not _is_file_scope(v[base].get("scope"))]
                cells = []
                for lo_w, hi_w in ((1, 1), (2, 2), (3, 4), (5, 99)):
                    p = [(d, c) for d, c in pairs
                         if lo_w <= len((d.get("query") or "").split()) <= hi_w]
                    if len(p) < 10:
                        cells.append(f"{'n<10':>10}"); continue
                    d_ = (sum(hit(c) for _, c in p) - sum(hit(d) for d, _ in p)) / len(p)
                    cells.append(f"{d_:>+7.3f}({len(p):>3})".rjust(10))
                print(f"  {cand:<17}" + "".join(cells))


if __name__ == "__main__":
    main()
