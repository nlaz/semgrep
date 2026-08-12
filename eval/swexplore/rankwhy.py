#!/usr/bin/env python3
"""Subclassify the ranking bucket: per region, which engine stage loses gold.

For each replayable NEVER_SURFACED_W region, rerun the session's wide queries
at k=30 under five configurations on the same checkout:

    hybrid   — campaign flags (--chunking function), the baseline
    bm25     — --mode bm25 (lexical channel alone)
    sem      — --mode semantic (embedding channel alone)
    nofine   — hybrid --no-fine (coarse chunk order, no fine rerank)
    window   — hybrid with default window-32 chunking (no --chunking function)

plus two probes independent of the agent's wording:

    self     — query = identifier tokens drawn from the gold region's own text
               (semantic mode). If this misses, the pipeline cannot find the
               chunk from its own content — an indexing/chunking/embedding
               defect, not a vocabulary gap.
    exact    — sg -e -F <a literal line from the gold region>. If this misses,
               the file is not searchable at all (walk/filter exclusion).

Also records the top-5 files under the campaign config, to characterize what
outranked gold. Caches are wiped per repo; both chunkings are built.

    python3 eval/swexplore/rankwhy.py --misswhy <misswhy --json output> \
        --out rankwhy-results.jsonl

Classification of the output (RESEARCH.md §32.4a): exact probe missing ->
file not searchable; self probe missing -> gold text too generic to rank;
hybrid <=30 -> in-pool ordering; else the first variant whose top-5 has it
names the stage that lost it (bm25 -> fusion drowned a lexical hit, nofine ->
fine rerank killed it, window -> function chunking did); none -> vocab gap.
"""
import argparse, json, os, pathlib, re, shutil, subprocess, collections, time, tempfile

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
SG = ROOT / "target/release/sg"
DATA = ROOT / "eval/data/swexplore"
K = 30

ap = argparse.ArgumentParser()
ap.add_argument("--misswhy", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--run-id", default="s32")
ap.add_argument("--arm", default="sub-sg")
args = ap.parse_args()
PATHLINE = re.compile(r"^([^\s:][^:]*):(\d+):", re.M)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

CONFIGS = {
    "hybrid": ["--chunking", "function"],
    "bm25":   ["--chunking", "function", "--mode", "bm25"],
    "sem":    ["--chunking", "function", "--mode", "semantic"],
    "nofine": ["--chunking", "function", "--no-fine"],
    "window": [],
}

def repo_root(iid):
    t = DATA / "runs" / args.run_id / iid / args.arm / "trace.jsonl"
    if not t.exists():
        return None
    for line in open(t):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rc = (e.get("input") or {}).get("root_canonical")
        if rc:
            return rc
    return None

def groups_of(stdout):
    out = []
    for block in stdout.split("\n\n"):
        files = {m.group(1) for m in PATHLINE.finditer(block)}
        if files:
            out.append(files)
    return out

def rank_of(groups, path):
    return next((i + 1 for i, fs in enumerate(groups) if path in fs), None)

def gold_text(repo, gp, s, e):
    try:
        lines = (pathlib.Path(repo) / gp).read_text(errors="replace").splitlines()
    except OSError:
        return None, None
    lo, hi = max(1, s), min(len(lines), e)
    seg = lines[lo - 1:hi]
    toks = []
    for ln in seg:
        toks += IDENT.findall(ln)
    # widen a thin region until we have enough identifiers to embed
    pad = 1
    while len(toks) < 4 and (lo - pad >= 1 or hi + pad <= len(lines)) and pad < 12:
        extra = lines[max(0, lo - 1 - pad):lo - 1] + lines[hi:hi + pad]
        toks = [t for ln in extra for t in IDENT.findall(ln)] + toks
        pad += 1
    # most distinctive line = the longest; 'use strict'-style openers match
    # half the repo and the 250-match display cap then hides gold
    cands = [ln.strip() for ln in seg if len(ln.strip()) >= 12 and IDENT.search(ln)]
    literal = max(cands, key=len) if cands else None
    seen, q = set(), []
    for t in toks:
        if t.lower() not in seen:
            seen.add(t.lower()); q.append(t)
        if len(q) >= 12:
            break
    return (" ".join(q) if q else None), literal

rows = [json.loads(l) for l in open(args.misswhy)]
rows = [r for r in rows if r["bucket"] == "NEVER_SURFACED_W" and r["wide_queries"]]
by_repo = collections.defaultdict(list)
for r in rows:
    root = repo_root(r["instance_id"])
    if root and pathlib.Path(root).is_dir():
        by_repo[root].append(r)

print(f"{sum(len(v) for v in by_repo.values())} regions across {len(by_repo)} repos", flush=True)
cache = pathlib.Path(tempfile.mkdtemp(prefix="rankwhy-"))
out_f = open(args.out, "w")
t0 = time.time(); n = 0
try:
    for repo, rrows in sorted(by_repo.items()):
        env = dict(os.environ, SEMGREP_CACHE_DIR=str(cache))
        def run(argv):
            p = subprocess.run([str(SG), *argv], cwd=repo, env=env,
                               capture_output=True, text=True, timeout=900)
            return p.stdout, p.returncode
        qcache = {}
        queries = sorted({q for r in rrows for q in r["wide_queries"]})
        for q in queries:
            per = {}
            for name, flags in CONFIGS.items():
                so, rc = run([q, *flags, "-k", str(K)])
                per[name] = groups_of(so)
            qcache[q] = per
        for r in rrows:
            gp, s, e = r["region"]
            rec = {"instance_id": r["instance_id"], "region": r["region"],
                   "queries": {}}
            best = {name: None for name in CONFIGS}
            top5_beat = None
            for q in r["wide_queries"]:
                per = qcache[q]
                qranks = {name: rank_of(per[name], gp) for name in CONFIGS}
                rec["queries"][q] = qranks
                for name, rk in qranks.items():
                    if rk and (best[name] is None or rk < best[name]):
                        best[name] = rk
                if top5_beat is None and per["hybrid"]:
                    top5_beat = [sorted(fs)[0] for fs in per["hybrid"][:5]]
            rec["best"] = best
            rec["top5_hybrid_q0"] = top5_beat
            selfq, literal = gold_text(repo, gp, s, e)
            if selfq:
                so, rc = run([selfq, "--chunking", "function",
                              "--mode", "semantic", "-k", "5"])
                rec["self_rank"] = rank_of(groups_of(so), gp)
                rec["self_query"] = selfq
            else:
                rec["self_rank"] = None
                rec["self_query"] = None
            if literal:
                so, rc = run(["-e", "-F", "-l", literal])
                rec["exact_found"] = gp in so.splitlines()
            else:
                rec["exact_found"] = None
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()
            n += 1
        for entry in cache.iterdir():
            shutil.rmtree(entry, ignore_errors=True)
        print(f"[{time.time()-t0:6.0f}s] {pathlib.Path(repo).name[:52]}: "
              f"{len(rrows)} regions ({n} total)", flush=True)
finally:
    shutil.rmtree(cache, ignore_errors=True)
    out_f.close()
print("DONE", n)
