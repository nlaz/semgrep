#!/usr/bin/env bash
# Fetch SWE-Explore (arXiv 2606.07297) into eval/data/swexplore/.
#
# Three artifacts, because the benchmark ships only two thirds of itself:
#
#   bench.jsonl   848 instances with line-level ground truth (from HF)
#   sidecar.jsonl the issue text, base_commit and language per instance
#
# The sidecar exists because the HF dataset carries NO issue text. Upstream
# resolves it from a `unify_trajs/` directory that is not in the GitHub repo
# and not published anywhere we can find. Rebuilding it from the three
# benchmarks the instances were drawn from is not a workaround — it is
# strictly better provenance: the issue text comes from the canonical
# upstream set rather than from somebody's trajectory dump.
#
# Verified 451 + Multilingual 182 + Pro 215 = 848. The script hard-fails on
# anything less, because a partial map silently drops instances from a frame
# and reads as "those instances scored zero".
#
# Re-runnable; skips work already done. Needs network + uv.
set -euo pipefail
cd "$(dirname "$0")"

OUT=../data/swexplore
mkdir -p "$OUT"

# ---------------------------------------------------------------- harness
# Upstream pinned at its only commit. Our changes are applied from checked-in
# files rather than committed into a vendored copy, so `git diff` inside
# upstream/ always shows exactly what we changed and nothing else — and a
# future upstream commit makes the patch fail loudly instead of silently
# applying to code that moved.
UPSTREAM_SHA=3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1
if [ ! -d "$OUT/upstream/.git" ]; then
  echo "cloning SWE-Explore-Bench at $UPSTREAM_SHA ..."
  git clone -q https://github.com/Qiushao-E/SWE-Explore-Bench.git "$OUT/upstream"
fi
git -C "$OUT/upstream" checkout -q "$UPSTREAM_SHA"
git -C "$OUT/upstream" checkout -q -- .          # drop any prior patch
cp patches/_sg_repos.py "$OUT/upstream/_sg_repos.py"
cp patches/sg_arms.py patches/sg_static.py "$OUT/upstream/explorers/"
git -C "$OUT/upstream" apply ../../../swexplore/patches/0001-eval_runner-arms-cost-rolling.patch
python3 -m py_compile "$OUT/upstream/eval_runner.py" "$OUT/upstream/explorers/sg_arms.py" \
                      "$OUT/upstream/explorers/sg_static.py" "$OUT/upstream/_sg_repos.py"
echo "harness ready: upstream@${UPSTREAM_SHA:0:7} + 3 files + 1 patch (eval.py untouched)"

# ---------------------------------------------------------------- dataset
if [ -s "$OUT/bench.jsonl" ] && [ -s "$OUT/sidecar.jsonl" ]; then
  echo "dataset already fetched: $(wc -l < "$OUT/bench.jsonl" | tr -d ' ') bench rows, \
$(wc -l < "$OUT/sidecar.jsonl" | tr -d ' ') sidecar rows"
  exit 0
fi

uv run --with datasets python - <<'PY'
from datasets import load_dataset
import collections, hashlib, json, pathlib, sys

OUT = pathlib.Path("../data/swexplore")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- bench
sx = load_dataset("SWE-Explore-Bench/SWE-Explore-Bench", split="train")
with (OUT / "bench.jsonl").open("w") as f:
    for r in sx:
        f.write(json.dumps(r) + "\n")
print(f"bench.jsonl: {len(sx)} rows")

want = {iid: d for iid, d in zip(sx["instance_id"], sx["dataset"])}

# ------------------------------------------------------- issue + commit
# `norm` strips SWE-bench Pro's "instance_" prefix. Pro is the only source
# that carries one, and without this every one of its 215 rows misses.
def norm(iid):
    return iid[len("instance_"):] if iid.startswith("instance_") else iid

SOURCES = [
    ("princeton-nlp/SWE-bench_Verified", "test"),
    ("SWE-bench/SWE-bench_Multilingual", "test"),
    ("ScaleAI/SWE-bench_Pro", "test"),
]

side, provenance = {}, {}
for name, split in SOURCES:
    d = load_dataset(name, split=split)
    new = 0
    for iid, ps, bc, repo in zip(d["instance_id"], d["problem_statement"],
                                 d["base_commit"], d["repo"]):
        k = norm(iid)
        if k in want and k not in side:
            side[k] = {"problem_statement": ps, "base_commit": bc, "repo": repo}
            provenance[k] = name
            new += 1
    print(f"  {name}: n={len(d)} -> matched {new} (cumulative {len(side)}/{len(want)})")

missing = sorted(set(want) - set(side))
if missing:
    print(f"\nFATAL: {len(missing)} instances have no issue text; "
          f"by split {collections.Counter(want[i] for i in missing)}", file=sys.stderr)
    print(f"first few: {missing[:5]}", file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------- language
# Inferred from the gold files, not from the `dataset` split: `pro` mixes
# Python and non-Python (96/119), so the split is not a language label.
# Reproduces the paper's Table 7 on the languages where the mapping is
# unambiguous (Python 547, Go 84, Rust 31, Java 30, PHP 28, Ruby 22 all
# exact); .h and .jsx/.tsx are judgement calls and ours differ slightly.
EXT_LANG = {
    ".py": "Python", ".go": "Go", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".rs": "Rust", ".java": "Java",
    ".php": "PHP", ".rb": "Ruby", ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".hpp": "C++",
}

def language_of(gold_files):
    c = collections.Counter()
    for p in gold_files or []:
        ext = "." + p.rsplit(".", 1)[-1] if "." in p else ""
        if ext in EXT_LANG:
            c[EXT_LANG[ext]] += 1
    return c.most_common(1)[0][0] if c else "other"

with (OUT / "sidecar.jsonl").open("w") as f:
    for r in sx:
        iid = r["instance_id"]
        gt = r["ground_truth"] or {}
        f.write(json.dumps({
            "instance_id": iid,
            "dataset": r["dataset"],
            "language": language_of(gt.get("read_core_files")),
            # `repo` and `base_commit` come from the upstream source, not
            # parsed out of instance_id — Pro's ids are org__repo-<sha>-v<sha>
            # and Verified's are org__repo-<PR#>, so any parser needs two
            # cases and gets repos with '-' in the name wrong.
            "repo": side[iid]["repo"],
            "base_commit": side[iid]["base_commit"],
            "problem_statement": side[iid]["problem_statement"],
            "issue_source": provenance[iid],
            "repo_dir": r["repo_dir"],
        }) + "\n")

# eval_runner's --issue-map wants {instance_id: issue_text}; its only other
# source is rec["problem_statement"], which bench.jsonl does not carry.
(OUT / "issue_map.json").write_text(json.dumps(
    {k: v["problem_statement"] for k, v in side.items()}))

langs = collections.Counter(json.loads(l)["language"]
                            for l in (OUT / "sidecar.jsonl").read_text().splitlines())
print(f"\nsidecar.jsonl: {len(side)} rows")
print("language:", dict(langs.most_common()))

for name in ("bench.jsonl", "sidecar.jsonl"):
    h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()[:16]
    print(f"  {name} sha256={h}")
PY
