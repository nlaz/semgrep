"""Rolling checkouts and persisted line counts for the §27 campaign.

Dropped into the upstream tree by `fetch.sh`. Two jobs, both forced by
measurements rather than preference:

**Rolling checkouts.** Upstream assumes `--repos` is fully prefetched. Measured
over 19 instances spanning all ten languages, a checkout averages 32.1 MB
(1.2 MB axum → 113 MB teleport), so 848 of them is **26.6 GB** — against 21 GiB
free on this machine, before semgrep indexes, which live inside the checkout
and roughly double it. So checkouts are fetched on demand and evicted under a
byte cap.

A byte-capped LRU rather than refcounting, because `eval_runner.py`'s loop is
**explorer-major** (`for name in explorer_names:` at :597, instances inside).
Each instance is therefore visited once per arm in three separate passes, and
a refcount that freed after the last arm would free after pass three — i.e.
never, during pass one. An LRU degrades gracefully instead: whatever fits is
reused across passes, the rest is re-fetched at ~8 MB and 1–3 s a time, which
is nothing against a 600 s agent run.

**Persisted line counts.** This is a correctness fix, not a disk one.
`eval.py:_resolve_interval` returns `None` — silently dropping the region from
every metric — for any gold region with `end == -1` or `start < 0` when the
file's line count is unknown. Upstream's `_build_file_line_counts` walks every
record up front and so needs every repo present at once, which eviction makes
impossible. Counts are therefore taken while each checkout is live and appended
to `file_line_counts.json`, which survives the eviction that follows.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

# Bytes of checkouts to keep resident. 8 GB holds ~250 of the mean 32 MB
# checkout, so most of a pass is reused rather than re-fetched.
CAP_BYTES = int(float(os.environ.get("SWEXPLORE_CACHE_GB", "8")) * 1024**3)

_lock = threading.Lock()
_lru: dict[str, float] = {}      # instance_id -> last-used monotonic time
_sizes: dict[str, int] = {}      # instance_id -> bytes on disk
_sidecar: dict[str, dict] | None = None
_bootstrapped = False
_inflight: dict[str, int] = {}   # instance_id -> workers currently inside it

# Backstop if a release() is ever missed on an exception path: nothing touched
# within this window is evictable regardless. Sized above the 600 s agent
# timeout plus index build.
PROTECT_SECS = 900.0


def acquire(instance_id: str) -> None:
    with _lock:
        _inflight[instance_id] = _inflight.get(instance_id, 0) + 1


def release(instance_id: str) -> None:
    with _lock:
        n = _inflight.get(instance_id, 0) - 1
        if n > 0:
            _inflight[instance_id] = n
        else:
            _inflight.pop(instance_id, None)


def _bootstrap_locked(repos_root: Path) -> None:
    """Register checkouts already on disk from an earlier rung.

    Without this the evictor only knows what *this process* fetched, and a
    resumed instance never calls `_get_repo_dir` at all — so a later rung
    inherits the previous rung's checkouts as an invisible floor it can never
    reclaim. Measured live during R2: 215 checkouts and 9.0 GB with the LRU
    reporting well under its 5 GB cap and never having evicted once, heading
    for ~14 GB against 13 GB free.

    Seeded with timestamps *older* than anything this process will touch, so
    the inherited set is evicted first — it is by definition the least
    recently useful.
    """
    global _bootstrapped
    if _bootstrapped or not repos_root.is_dir():
        _bootstrapped = True
        return
    for i, d in enumerate(sorted(p for p in repos_root.iterdir() if p.is_dir())):
        if d.name not in _sizes:
            _sizes[d.name] = _du(d)
            _lru[d.name] = -1e9 + i     # older than any time.monotonic()
    _bootstrapped = True


def _load_sidecar(data_dir: Path) -> dict[str, dict]:
    """instance_id -> {repo, base_commit, ...} from fetch.sh's sidecar."""
    global _sidecar
    if _sidecar is None:
        _sidecar = {}
        p = data_dir / "sidecar.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                _sidecar[r["instance_id"]] = r
    return _sidecar


def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _download(repo: str, commit: str, dest: Path) -> bool:
    """GitHub archive tarball -> dest. Same endpoint upstream's fetch_repos
    uses, so the tree is byte-identical to what their published run saw."""
    url = f"https://github.com/{repo}/archive/{commit}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / "a.tar.gz"
        # Retry with backoff. The pilot lost 29 of 31 instances in its first
        # pass to transient archive-API failures under 4 workers, and the
        # loss was *silent* — ensure() returned None, the runner skipped the
        # instance, and the arm simply came back short. Silent skips are the
        # worst failure mode here: they cost money, and they choose which
        # instances get measured.
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=300) as r, tgz.open("wb") as f:
                    shutil.copyfileobj(r, f)
                break
            except Exception:
                if attempt == 3:
                    return False
                time.sleep(2 ** attempt * 3)
        stage = Path(td) / "x"
        stage.mkdir()
        try:
            with tarfile.open(tgz, "r:gz") as tar:
                members = tar.getmembers()
                if not members:
                    return False
                top = members[0].name.split("/")[0]
                tar.extractall(stage)
        except Exception:
            return False
        src = stage / top
        if not src.is_dir():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Move onto the final name last, so a crash mid-extract cannot leave a
        # half-tree that the next run mistakes for a complete checkout.
        shutil.move(str(src), str(dest))
    return True


def _evict_locked(repos_root: Path, keep: str) -> None:
    total = sum(_sizes.values())
    if total <= CAP_BYTES:
        return
    freed, n, held = 0, 0, 0
    now = time.monotonic()
    for iid in sorted(_lru, key=lambda k: _lru[k]):
        if total <= CAP_BYTES:
            break
        if iid == keep:
            continue
        # NEVER evict a checkout another worker is inside. The first version
        # protected only `keep` — the instance this call is ensuring — so with
        # 5 workers one thread would delete the working directory of an agent
        # running in another. Measured: 432 of 848 cc-sg rows died at 2.7 s
        # with 1 turn and cwd gone. It hit cc-sg alone because that is the only
        # arm building an index, which inflates every tree ~30% and keeps the
        # cache permanently over its cap, so eviction ran constantly.
        if _inflight.get(iid) or (now - _lru[iid]) < PROTECT_SECS:
            held += 1
            continue
        d = repos_root / iid
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        sz = _sizes.pop(iid, 0)
        total -= sz
        freed += sz
        n += 1
        _lru.pop(iid, None)
    # Loud on purpose. The first version of this evicted nothing for 215
    # checkouts and 9 GB and said nothing about it; silence was why the bug
    # survived a whole rung.
    if n or total > CAP_BYTES:
        print(f"[sg_repos] evicted {n} checkout(s), freed {freed / 1e9:.2f} GB, "
              f"now {total / 1e9:.2f} GB of {CAP_BYTES / 1e9:.1f} GB"
              + (f", {held} held in-flight" if held else ""),
              file=sys.stderr, flush=True)


def ensure(instance_id: str, repos_root: Path, data_dir: Path) -> Path | None:
    """Return a live checkout for `instance_id`, fetching if needed."""
    dest = repos_root / instance_id
    with _lock:
        _bootstrap_locked(repos_root)
        if dest.is_dir():
            _lru[instance_id] = time.monotonic()
            # Re-measure rather than setdefault: `.semgrep/` is built *after*
            # the checkout is fetched and is 25-35% of the tree (2.1 GB of 8.3
            # GB measured live), so the fetch-time figure understates every
            # instance and the cap binds well above where it was set.
            _sizes[instance_id] = _du(dest)
            _evict_locked(repos_root, keep=instance_id)
            return dest
    side = _load_sidecar(data_dir).get(instance_id)
    if not side:
        return None
    ok = _download(side["repo"], side["base_commit"], dest)
    if not ok or not dest.is_dir():
        return None
    with _lock:
        _bootstrap_locked(repos_root)
        _lru[instance_id] = time.monotonic()
        _sizes[instance_id] = _du(dest)
        _evict_locked(repos_root, keep=instance_id)
    return dest


# ------------------------------------------------------------------ counts

def record_line_counts(instance_id: str, gt: dict, repo_dir: Path,
                       data_dir: Path) -> dict[str, int]:
    """Count lines for every gold path while the checkout is live, and append
    to file_line_counts.json. Without this, `end == -1` regions score as if
    they did not exist."""
    paths = set()
    for r in gt.get("read_core_regions") or []:
        if isinstance(r.get("path"), str):
            paths.add(r["path"])
    for regions in (gt.get("read_optional_regions_map") or {}).values():
        for r in regions:
            if isinstance(r.get("path"), str):
                paths.add(r["path"])
    per: dict[str, int] = {}
    for rel in paths:
        f = repo_dir / rel
        if f.is_file():
            try:
                per[rel] = len(f.read_text(errors="ignore").splitlines())
            except OSError:
                pass
    if per:
        with _lock:
            store = data_dir / "file_line_counts.json"
            cur = json.loads(store.read_text()) if store.exists() else {}
            cur[instance_id] = per
            tmp = store.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cur))
            tmp.replace(store)
    return per


def load_line_counts(data_dir: Path) -> dict[str, dict[str, int]]:
    store = data_dir / "file_line_counts.json"
    return json.loads(store.read_text()) if store.exists() else {}
