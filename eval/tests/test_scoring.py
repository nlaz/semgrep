"""Tests for the Loc-Bench scorer.

Every accuracy number in RESEARCH.md §11 comes out of `scoring.py`, and it had
no tests. A wrong `func_match` does not fail loudly — it shifts every conclusion
by a few points and looks like a result.

The cases here are the ones where a scorer is tempted to over-credit: an agent
that answers a bare method name for a qualified gold symbol, that answers a
path with a leading directory dropped, or that puts a file path in the functions
list. Over-crediting is the failure that matters, because it flatters the tool
under test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "locbench"))

import scoring  # noqa: E402


# ---------------------------------------------------------------------------
# gold parsing
# ---------------------------------------------------------------------------


def test_gold_splits_files_from_functions():
    files, funcs = scoring.parse_gold(
        ["src/a.py:Class.method", "src/a.py:other", "docs/b.md"]
    )
    assert files == ["src/a.py", "docs/b.md"], "each file listed once, in order"
    assert funcs == [("src/a.py", "Class.method"), ("src/a.py", "other")]


def test_gold_normalizes_paths():
    files, funcs = scoring.parse_gold(["./src/a.py:f", "src\\b.py:g"])
    assert files == ["src/a.py", "src/b.py"], "leading ./ and backslashes normalized"
    assert funcs[1][0] == "src/b.py"


def test_gold_of_nothing_is_empty():
    assert scoring.parse_gold([]) == ([], [])


# ---------------------------------------------------------------------------
# file matching
# ---------------------------------------------------------------------------


def test_file_match_prefers_exact():
    assert scoring.file_match("src/a.py", ["src/a.py", "other/a.py"]) == "src/a.py"


def test_file_match_accepts_a_unique_suffix():
    # The agent dropped a leading directory. Unambiguous, so credit it.
    assert scoring.file_match("a.py", ["src/a.py"]) == "src/a.py"
    assert scoring.file_match("src/a.py", ["repo/src/a.py"]) == "repo/src/a.py"


def test_file_match_refuses_an_ambiguous_suffix():
    # Two gold files could match, so crediting either would be a coin flip.
    assert scoring.file_match("a.py", ["src/a.py", "test/a.py"]) is None


def test_file_match_does_not_credit_a_substring():
    # "b.py" must not match "sub.py" — the suffix rule is on path segments.
    assert scoring.file_match("b.py", ["src/sub.py"]) is None


def test_file_match_of_nothing_is_none():
    assert scoring.file_match("a.py", []) is None


# ---------------------------------------------------------------------------
# function matching, and the tolerant rules specifically
# ---------------------------------------------------------------------------


def test_strict_matching_requires_exact_qualnames():
    assert scoring.func_match("Class.method", "Class.method", tolerant=False)
    assert not scoring.func_match("method", "Class.method", tolerant=False)
    assert not scoring.func_match("Other.method", "Class.method", tolerant=False)


def test_tolerant_matching_allows_a_dot_suffix_either_way():
    assert scoring.func_match("method", "Class.method", tolerant=True)
    assert scoring.func_match("Class.method", "method", tolerant=True)


def test_tolerant_matching_allows_leaf_equality():
    # This is the loosest rule and the one that can over-credit: two different
    # classes with a same-named method are indistinguishable to it. Asserted so
    # the looseness is a decision on record rather than a surprise.
    assert scoring.func_match("A.run", "B.run", tolerant=True)
    assert not scoring.func_match("A.run", "B.walk", tolerant=True)


def test_tolerant_matching_still_rejects_unrelated_names():
    assert not scoring.func_match("parse", "render", tolerant=True)


# ---------------------------------------------------------------------------
# prediction parsing
# ---------------------------------------------------------------------------


def test_malformed_predictions_are_counted_not_matched():
    preds, malformed = scoring.parse_pred_functions(
        ["src/a.py:f", "src/b.py", "src/c.py:g"]
    )
    assert preds == [("src/a.py", "f"), ("src/c.py", "g")]
    assert malformed == 1, "a functions entry without ':' is noise, and is counted"


def test_worktree_prefix_is_stripped():
    preds, _ = scoring.parse_pred_functions(
        ["/tmp/wt/src/a.py:f"], worktree_prefix="/tmp/wt"
    )
    assert preds == [("src/a.py", "f")]


def test_non_string_predictions_do_not_crash():
    # Agents have returned numbers and nulls here.
    preds, malformed = scoring.parse_pred_functions([42, None, "src/a.py:f"])
    assert preds == [("src/a.py", "f")]
    assert malformed == 2


# ---------------------------------------------------------------------------
# instance scoring: Acc@k means *complete* localization
# ---------------------------------------------------------------------------


def test_file_acc_requires_every_gold_file():
    gold_files, gold_funcs = scoring.parse_gold(["src/a.py:f", "src/b.py:g"])

    # Only one of two gold files predicted: not complete localization.
    partial = scoring.score_instance(["src/a.py"], [], gold_files, gold_funcs)
    assert partial["file_acc@1"] == 0
    assert partial["file_acc@5"] == 0
    assert partial["file_recall@5"] == 0.5

    both = scoring.score_instance(["src/a.py", "src/b.py"], [], gold_files, gold_funcs)
    assert both["file_acc@5"] == 1
    assert both["file_acc@1"] == 0, "two files cannot be covered by one prediction"
    assert both["file_recall@5"] == 1.0


def test_k_is_a_cutoff_on_the_prediction_list():
    gold_files, gold_funcs = scoring.parse_gold(["src/b.py:g"])
    # The gold file is predicted, but fourth — inside @5, outside @3 and @1.
    preds = ["x.py", "y.py", "z.py", "src/b.py"]
    m = scoring.score_instance(preds, [], gold_files, gold_funcs)
    assert (m["file_acc@1"], m["file_acc@3"], m["file_acc@5"]) == (0, 0, 1)


def test_function_acc_is_reported_at_both_strictnesses():
    gold_files, gold_funcs = scoring.parse_gold(["src/a.py:Class.method"])
    # A bare leaf name: tolerant credits it, strict does not.
    m = scoring.score_instance(["src/a.py"], ["src/a.py:method"], gold_files, gold_funcs)
    assert m["func_acc@5_strict"] == 0
    assert m["func_acc@5_tol"] == 1
    assert m["func_recall@10_strict"] == 0.0
    assert m["func_recall@10_tol"] == 1.0


def test_one_prediction_cannot_cover_two_gold_functions():
    gold_files, gold_funcs = scoring.parse_gold(
        ["src/a.py:Class.first", "src/a.py:Class.second"]
    )
    m = scoring.score_instance(
        ["src/a.py"], ["src/a.py:Class.first"], gold_files, gold_funcs
    )
    assert m["func_acc@10_tol"] == 0, "half the gold functions is not complete"
    assert m["func_recall@10_tol"] == 0.5


def test_a_duplicated_prediction_does_not_double_credit():
    gold_files, gold_funcs = scoring.parse_gold(
        ["src/a.py:first", "src/a.py:second"]
    )
    m = scoring.score_instance(
        ["src/a.py"],
        ["src/a.py:first", "src/a.py:first", "src/a.py:first"],
        gold_files,
        gold_funcs,
    )
    assert m["func_recall@10_tol"] == 0.5, "repeating one answer must not cover two"


def test_empty_gold_scores_zero_not_one():
    # An instance with no gold data must not read as a perfect score, or it
    # silently inflates every average it appears in.
    m = scoring.score_instance(["src/a.py"], ["src/a.py:f"], [], [])
    assert m["file_acc@5"] == 0
    assert m["func_acc@10_tol"] == 0
    assert m["file_recall@5"] is None, "recall is undefined, not 0 or 1"
    assert m["func_recall@10_tol"] is None


def test_no_predictions_scores_zero():
    gold_files, gold_funcs = scoring.parse_gold(["src/a.py:f"])
    m = scoring.score_instance([], [], gold_files, gold_funcs)
    assert m["file_acc@5"] == 0
    assert m["func_acc@10_tol"] == 0
    assert m["file_recall@5"] == 0.0
