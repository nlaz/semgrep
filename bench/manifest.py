#!/usr/bin/env python3
"""Record what a benchmark corpus actually is, so a number can be tied to it.

Every retrieval and perf figure this project publishes was measured against a
tree in `bench/corpora/`, and until now nothing recorded which tree. Only the
kernel was pinned (`linux-6.9.tar.xz`). `fetch-corpora.sh` cloned vscode at
`--depth 1` of HEAD and then deleted `.git`, and wikipedia from
`simplewiki-latest` — so both moved every time anyone refetched, silently, and
the vscode corpus on this disk has **no recoverable revision at all**.

Two different things are needed and only one of them can be recovered:

  - `revision` — what to fetch to get this tree back. Going forward
    `fetch-corpora.sh` pins it. For the trees already on disk it is unknown,
    and this records it as unknown rather than inventing one.
  - `tree_digest` — what this tree *is*. Computable from disk right now, for
    every corpus, including the unpinned ones. This is the field that actually
    guards a comparison: if the digest changes, the corpus changed, whether or
    not anyone knows which revision either one was.

So a result measured today against unpinned vscode is still checkable — not
reproducible from the internet, but detectably the same tree or not.

Usage:
  python3 bench/manifest.py                 # write/refresh bench/corpora/MANIFEST.json
  python3 bench/manifest.py --check         # fail if any tree differs from the manifest
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPORA = HERE / "corpora"
MANIFEST = CORPORA / "MANIFEST.json"

# Recorded per corpus by fetch-corpora.sh. "unknown" is a real, honest value:
# the tree predates pinning and its revision cannot be recovered.
SOURCES = {
    "linux": {"source": "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.9.tar.xz",
              "revision": "v6.9"},
    "vscode": {"source": "https://github.com/microsoft/vscode.git",
               "revision": "unknown (cloned --depth 1 of HEAD before pinning; "
                           ".git was deleted, so it cannot be recovered)"},
    "wikipedia": {"source": "https://dumps.wikimedia.org/simplewiki/latest/"
                            "simplewiki-latest-pages-articles.xml.bz2",
                  "revision": "unknown (fetched from 'latest' before pinning)"},
}

SKIP_DIRS = {".git", ".semgrep", "__pycache__"}


def tree_digest(root):
    """sha256 over the sorted (relpath, size, content-sha) listing.

    Sorted, so filesystem walk order cannot change it. Content-hashed rather
    than mtime/size-only, because an edit that preserves length is exactly the
    drift a cheap check would miss.
    """
    h = hashlib.sha256()
    n_files = total = 0
    entries = []
    for p in sorted(root.rglob("*")):
        if any(d in SKIP_DIRS for d in p.parts):
            continue
        if not p.is_file() or p.is_symlink():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        entries.append((str(p.relative_to(root)), len(data),
                        hashlib.sha256(data).hexdigest()))
        n_files += 1
        total += len(data)
    for rel, size, sha in entries:
        h.update(f"{rel}\0{size}\0{sha}\n".encode())
    return h.hexdigest()[:32], n_files, total


def build(names=None):
    out = {}
    if MANIFEST.exists():
        out = json.loads(MANIFEST.read_text())
    for d in sorted(CORPORA.iterdir()):
        if not d.is_dir() or (names and d.name not in names):
            continue
        t0 = time.time()
        digest, n_files, total = tree_digest(d)
        meta = dict(SOURCES.get(d.name, {"source": "unknown", "revision": "unknown"}))
        meta.update({
            "tree_digest": digest,
            "file_count": n_files,
            "total_bytes": total,
            "digested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        out[d.name] = meta
        print(f"{d.name:12s} {n_files:>7d} files  {total / 1e6:>8.1f} MB  "
              f"{digest[:16]}  ({time.time() - t0:.1f}s)")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def check():
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}; run without --check first")
        return 2
    want = json.loads(MANIFEST.read_text())
    bad = 0
    for name, meta in sorted(want.items()):
        d = CORPORA / name
        if not d.is_dir():
            print(f"{name:12s} MISSING from {CORPORA}")
            bad += 1
            continue
        digest, n_files, total = tree_digest(d)
        if digest != meta["tree_digest"]:
            print(f"{name:12s} CHANGED: {meta['tree_digest'][:16]} -> {digest[:16]} "
                  f"({meta['file_count']} -> {n_files} files)")
            print(f"             every number measured against {name} predates this tree")
            bad += 1
        else:
            print(f"{name:12s} ok ({n_files} files)")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify trees against the manifest instead of writing it")
    ap.add_argument("--only", default="", help="comma-separated corpus names")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or None
    sys.exit(check() if a.check else (build(names) and 0))
