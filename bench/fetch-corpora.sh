#!/usr/bin/env bash
# Fetch benchmark corpora into bench/corpora/ (~5-6 GB total).
# Each corpus is skipped if already present, so this is re-runnable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p corpora
cd corpora

# 1. Linux kernel source (~1.5 GB unpacked) — canonical grep benchmark corpus.
if [ ! -d linux ]; then
  echo "==> linux kernel"
  curl -fLO https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.9.tar.xz
  tar xf linux-6.9.tar.xz
  mv linux-6.9 linux
  rm linux-6.9.tar.xz
fi

# 2. VS Code (large TS/JS repo) — representative agent coding target.
if [ ! -d vscode ]; then
  echo "==> vscode"
  git clone --depth 1 https://github.com/microsoft/vscode.git vscode
  rm -rf vscode/.git   # benchmark the tree, not git history
fi

# 3. Wikipedia subset: Simple English Wikipedia, extracted to plain text
#    (~1 GB). Uses wikiextractor via uvx (pip install uv if missing).
if [ ! -d wikipedia ]; then
  echo "==> simple english wikipedia"
  curl -fLO https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
  uvx wikiextractor simplewiki-latest-pages-articles.xml.bz2 \
      --output wikipedia --bytes 256K --no-templates
  rm simplewiki-latest-pages-articles.xml.bz2
fi

echo "done:"
du -sh linux vscode wikipedia
