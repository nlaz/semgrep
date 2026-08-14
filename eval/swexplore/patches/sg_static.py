"""`sg` as a non-agentic explorer — the free arm.

Dropped into the upstream tree at `explorers/sg_static.py` by `fetch.sh`.
Scored beside their `bm25` / `tfidf` / `potion` baselines: no agent, no model,
no spend. It exists to answer the one question §26 could not — whether the
shipped 800-char passage budget behaves the same outside Python — across ten
languages for the price of some CPU.

**Regions are passage spans, not chunk spans.** `sg --json` gives both: the
chunk (`start_line`..`end_line`, 32 lines) and the passage actually printed
(`lines_from` .. `lines_from + len(lines) - 1`, ~17 lines at the shipped
800-char budget). The passage is what a caller is shown, so it is what the
agentic arms put in front of the model, and scoring the chunk instead would
credit `sg` for lines no agent ever saw. §24.1's lesson applies directly: the
two differ by a wide margin and reporting the flattering one is how a geometry
change gets published as a retrieval win. `SWEXPLORE_SG_SPAN=chunk` switches it
for the sensitivity check, which is the honest way to show the bracket.

Expect this to land in the classical-retrieval tier, well below the agentic
explorers. That is not a semgrep result: the gold is what a multi-turn agent
read across a whole session, and one top-5 query cannot substitute for a
session. The value here is the *cross-language shape*, not the rank.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import ContextRegion, ExplorerResult
from .sg_arms import SEMGREP_BIN

# The issue statement is used as the query verbatim, matching what upstream
# does for BM25 and TF-IDF ("use the issue statement as the query"). It is a
# poor query — median 1,212 chars against the ~5 words real agents type
# (RESEARCH.md §21.2) — but changing it here would make this arm
# incomparable to the baselines it sits beside in their table.
MAX_QUERY_CHARS = int(os.environ.get("SWEXPLORE_SG_QUERY_CHARS", "4000"))
SPAN = os.environ.get("SWEXPLORE_SG_SPAN", "passage")


@dataclass
class SgStaticExplorer:
    repo_root: Path
    timeout: int = 300

    def _index(self) -> None:
        idx = Path(self.repo_root) / ".semgrep" / "meta.json"
        if idx.exists() and idx.stat().st_mtime >= SEMGREP_BIN.stat().st_mtime:
            return
        subprocess.run([str(SEMGREP_BIN), "index", str(self.repo_root)],
                       capture_output=True, timeout=self.timeout)

    def explore(self, *, instance_id: str, query: str, top_k: int = 5
                ) -> List[ExplorerResult]:
        self._index()
        q = " ".join(query.split())[:MAX_QUERY_CHARS]
        # cwd is the repo root so `path` comes back repo-relative, which is
        # the frame SWE-Explore's gold is expressed in.
        p = subprocess.run(
            [str(SEMGREP_BIN), q, ".", "--json", "-k", str(top_k)],
            cwd=str(self.repo_root), capture_output=True, text=True,
            timeout=self.timeout,
            env={**os.environ, "SEMGREP_NO_HINTS": "1"},
        )
        regions: list[ContextRegion] = []
        for line in p.stdout.splitlines():
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            if SPAN == "chunk" or h.get("lines_from") is None:
                start, end = h["start_line"], h["end_line"]
            else:
                start = h["lines_from"]
                end = start + max(1, len(h.get("lines") or [1])) - 1
            regions.append(ContextRegion(path=h["path"], start=start, end=end))
        return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
