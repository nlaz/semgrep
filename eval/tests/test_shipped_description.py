"""The description README tells you to paste must be the one the harness scores.

README.md recommends a tool description to put in an agent's system prompt.
`eval/locbench/run.py` holds the same string as the `desc-v8` campaign arm.
Nothing but this test keeps them equal, and they are easy to drift apart: one
gets reworded for a reader, the other stays what was measured, and the README
then recommends a description no campaign has ever run.

That failure is silent and expensive — RESEARCH.md §16.6 measured a single
clause moving an agent's ranked share from 72% to 7%, so "close enough"
wording is not close enough to carry a measurement across.

The README copy legitimately drops the harness-only tail (`Read and Glob are
also available`), which describes the eval sandbox rather than the tool, and
legitimately re-wraps for width. Everything else must match token for token.
"""

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "locbench"))

HEADING = "### The tool description to give your agent"
# Present in the scored arm because the harness blocks every other tool; not
# part of what semgrep recommends to a reader, so the README may omit it.
HARNESS_ONLY = " Read and Glob are also available."


def normalize(s):
    """Collapse whitespace: the README wraps for width, the harness string does not."""
    return re.sub(r"\s+", " ", s).strip()


def readme_snippet():
    md = (ROOT / "README.md").read_text()
    assert HEADING in md, f"README lost the {HEADING!r} section"
    after = md[md.index(HEADING):]
    blocks = after.split("```")
    assert len(blocks) > 2, "no fenced block under the tool-description heading"
    return blocks[1]


def test_readme_matches_the_scored_desc_v8_arm():
    import run

    canonical = normalize(run.DESC_CONDITIONS["desc-v8"])
    expected = normalize(canonical.replace(HARNESS_ONLY, ""))
    assert normalize(readme_snippet()) == expected


def test_the_example_query_is_names_rather_than_a_question():
    """The whole point of desc-v8 over desc-v7 (§19.2b).

    Guards the one span that carries the finding: reworded back into a question
    the string would still read fine, still parse, and quietly recommend the
    style that finds a blind answer 13% of the time instead of 50%.
    """
    import run

    example = run.DESC_CONDITIONS["desc-v8"]
    quoted = re.findall(r'semgrep "([^"]+)"', example)
    assert quoted, "desc-v8 lost its worked example"
    query = quoted[-1]
    question_words = {"where", "how", "what", "does", "is", "the", "which", "why"}
    assert not (set(query.lower().split()) & question_words), (
        f"desc-v8's example query {query!r} reads as a description; §19.2b's "
        f"whole result is that it must demonstrate candidate names"
    )
    assert len(query.split()) > 1, "the example should show several candidate names"


def test_v7_and_v8_differ_only_inside_the_example():
    """The pair is a single-variable contrast; if it stops being one, so does §19.4."""
    import run

    v7, v8 = run.DESC_CONDITIONS["desc-v7"], run.DESC_CONDITIONS["desc-v8"]
    ops = [o for o in difflib.SequenceMatcher(None, v7, v8).get_opcodes() if o[0] != "equal"]
    assert len(ops) == 1, f"v7/v8 differ in {len(ops)} places, not 1: {ops}"
    tag, i1, i2, j1, j2 = ops[0]
    assert tag == "replace"
    # The changed span must be the example's query and nothing structural.
    assert "retry" in v7[i1:i2].lower() and "retry" in v8[j1:j2].lower()
