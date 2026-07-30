#!/usr/bin/env bash
# Fetch benchmark corpora into bench/corpora/ (~5-6 GB for the big three,
# +~75 MB for the language corpora). Each corpus is skipped if already
# present, so this is re-runnable.
#
# EVERY corpus is pinned. It did not used to be: vscode was cloned at
# `--depth 1` of HEAD and wikipedia from `simplewiki-latest`, so both moved
# under anyone who refetched, and every vscode/wikipedia number in RESEARCH.md
# was measured against a tree nobody can name. The trees already on disk keep
# `revision: unknown` in the manifest — that cannot be recovered and is not
# invented. What CAN be recovered is what the tree *is*, so run
#
#     python3 bench/manifest.py            # record every tree's digest
#     python3 bench/manifest.py --check    # detect any that has changed since
#
# and a comparison stays checkable whether or not its corpus was pinned.
#
# Pinning going forward does mean a fresh clone differs from the tree already
# on this disk. That is the honest trade: a detectable difference beats an
# undetectable one.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p corpora
cd corpora

VSCODE_SHA=db0569eab9d4a70ff7cd38b9013dd20b30a67207
TOKIO_SHA=adc2ae7af2caaea83985fbdfbc7884c159c486f2
OKHTTP_SHA=f258f7a14b3a56106ed051a841ce71fa8a978b97
ETCD_SHA=c1dc77f1da858ef50a847fd249c45bff43c1fe58
JEKYLL_SHA=7697d249793d6c48c66a7293310a718aec01f660

# Clone one commit and drop the history: we benchmark the tree, not the repo.
# --filter=blob:none keeps the fetch to the blobs this commit actually needs.
pin_clone() {   # pin_clone <dir> <url> <sha>
  local dir=$1 url=$2 sha=$3
  echo "==> $dir @ ${sha:0:12}"
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
  git -C "$dir" checkout -q FETCH_HEAD
  rm -rf "$dir/.git"
}

# --- the big three ----------------------------------------------------------

# 1. Linux kernel source (~1.5 GB unpacked) — canonical grep benchmark corpus.
#    The only corpus that was pinned from the start.
if [ ! -d linux ]; then
  echo "==> linux kernel v6.9"
  curl -fLO https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.9.tar.xz
  tar xf linux-6.9.tar.xz
  mv linux-6.9 linux
  rm linux-6.9.tar.xz
fi

# 2. VS Code (large TS/JS repo) — representative agent coding target.
[ -d vscode ] || pin_clone vscode https://github.com/microsoft/vscode.git "$VSCODE_SHA"

# 3. Wikipedia subset: Simple English Wikipedia, extracted to plain text
#    (~1 GB). Uses wikiextractor via uvx (pip install uv if missing).
#    Wikimedia expires dated dumps after a few months, so `latest` is the only
#    URL that keeps working — this one genuinely cannot be pinned upstream, and
#    the tree digest is the only guard available. Record it.
if [ ! -d wikipedia ]; then
  echo "==> simple english wikipedia (UNPINNABLE — dated dumps expire upstream)"
  curl -fLO https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
  uvx wikiextractor simplewiki-latest-pages-articles.xml.bz2 \
      --output wikipedia --bytes 256K --no-templates
  rm simplewiki-latest-pages-articles.xml.bz2
fi

# --- language corpora (~75 MB total) ----------------------------------------
#
# `eval/symbols.py` supports python/js/ts/rust/go/c/java/ruby, and until these
# landed only c and ts were exercised by any corpus — the go, java and ruby
# regexes were tested on hand-written fixtures alone. All four also sit in the
# <2k-file band where RESEARCH.md §9.7 found engine variants actually diverge;
# the big three are 84k, 4k and 1k files.

if [ "${1:-}" != "--big-three-only" ]; then
  [ -d tokio ]  || pin_clone tokio  https://github.com/tokio-rs/tokio.git "$TOKIO_SHA"
  [ -d okhttp ] || pin_clone okhttp https://github.com/square/okhttp.git  "$OKHTTP_SHA"
  [ -d jekyll ] || pin_clone jekyll https://github.com/jekyll/jekyll.git  "$JEKYLL_SHA"
  if [ ! -d etcd ]; then
    pin_clone etcd https://github.com/etcd-io/etcd.git "$ETCD_SHA"
    # Vendored deps are other projects' code: they trip the near-duplicate
    # detectors and inflate the corpus without adding etcd to it.
    rm -rf etcd/vendor
  fi
fi

echo "done:"
du -sh -- */ 2>/dev/null | sort -h

echo
echo "recording tree digests (a corpus with no pinned revision is still checkable)"
python3 ../manifest.py
