"""Tests for the agent-query replay scorer.

replay.py had never been run when these were written — `replay.jsonl` did not
exist — so nothing here is protecting an existing number. It is protecting the
first one, which is the cheaper time to do it.

Two of the four defects these cover would have produced a *quotable* wrong
result rather than an error: a naive bootstrap over queries that cluster hard
by instance reports a CI far tighter than the data supports, and a path-suffix
fallback credits a hit on a different file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "locbench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import replay  # noqa: E402


def hits(*paths):
    return [{"path": p, "start_line": 1, "end_line": 2} for p in paths]


# --- rank_of_gold ------------------------------------------------------------

def test_an_exact_path_match_credits():
    assert replay.rank_of_gold(hits("a.py", "src/b.py"), ["src/b.py"]) == 2


def test_a_miss_is_none():
    assert replay.rank_of_gold(hits("a.py"), ["src/b.py"]) is None


def test_the_rank_is_the_earliest_gold_not_the_first_gold_listed():
    # golds are sorted, hits are ranked; the answer is about the ranking.
    got = replay.rank_of_gold(hits("x.py", "z.py", "a.py"), ["a.py", "z.py"])
    assert got == 2


def test_a_suffix_only_match_does_not_credit():
    # `tests/test_a.py` is not `src/tests/test_a.py`. A suffix fallback used
    # to credit it; instrumented across 2,217 scored queries it fired 0 times,
    # so it was over-crediting that never got the chance, and it is gone.
    assert replay.rank_of_gold(hits("tests/test_a.py"), ["src/tests/test_a.py"]) is None
    assert replay.rank_of_gold(hits("src/tests/test_a.py"), ["tests/test_a.py"]) is None


def test_a_hit_with_no_path_key_does_not_crash():
    assert replay.rank_of_gold([{"start_line": 1}], ["a.py"]) is None


# --- parse_argv --------------------------------------------------------------

def test_a_valued_flag_is_not_mistaken_for_the_query():
    assert replay.parse_argv(["-k", "20", "the query"])[1] == "the query"


def test_exact_mode_is_detected():
    is_exact, q, _ = replay.parse_argv(["-e", "def persist"])
    assert is_exact and q == "def persist"


def test_paths_after_the_query_are_scopes_not_the_query():
    _, q, scopes = replay.parse_argv(["my query", "src/", "tests/"])
    assert q == "my query" and scopes == ["src/", "tests/"]


# --- harvest -----------------------------------------------------------------

def log_dir(tmp_path, instance, condition, entries):
    import json
    d = tmp_path / instance / condition
    d.mkdir(parents=True)
    (d / "shim_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n")
    return tmp_path


def test_harvest_excludes_rg_by_default():
    # The shim logs hold 570 rg calls and they are regexes, not queries.
    # Replaying `csrf|CSRF|X-CSRF|wtf` through a ranked engine measures
    # punctuation tokenization, not retrieval.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = log_dir(Path(td), "inst1", "rg", [
            {"tool": "rg", "blocked": False, "argv": ["csrf|CSRF"]},
            {"tool": "semgrep", "blocked": False, "argv": ["auth flow"]},
        ])
        got = replay.harvest(root, want_exact=False)
        assert [q["query"] for q in got] == ["auth flow"]


def test_harvest_can_be_widened_to_rg():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = log_dir(Path(td), "inst1", "rg", [
            {"tool": "rg", "blocked": False, "argv": ["csrf|CSRF"]},
        ])
        got = replay.harvest(root, want_exact=False, tools=("rg",))
        assert [q["query"] for q in got] == ["csrf|CSRF"]


def test_harvest_skips_blocked_calls():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = log_dir(Path(td), "inst1", "sg", [
            {"tool": "semgrep", "blocked": True, "argv": ["nope"]},
        ])
        assert replay.harvest(root, want_exact=False) == []


def test_harvest_records_instance_and_condition():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = log_dir(Path(td), "OWNER__repo-1", "sg-plain", [
            {"tool": "semgrep", "blocked": False, "argv": ["q"]},
        ])
        got = replay.harvest(root, want_exact=False)
        assert got[0]["instance_id"] == "OWNER__repo-1"
        assert got[0]["condition"] == "sg-plain"


def test_harvest_separates_exact_from_ranked():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = log_dir(Path(td), "i", "c", [
            {"tool": "semgrep", "blocked": False, "argv": ["-e", "tok"]},
            {"tool": "semgrep", "blocked": False, "argv": ["words here"]},
        ])
        assert [q["query"] for q in replay.harvest(root, False)] == ["words here"]
        assert [q["query"] for q in replay.harvest(root, True)] == ["tok"]


# --- index-flag guard --------------------------------------------------------

def test_an_index_affecting_flag_is_rejected():
    # Replay shares one index per worktree, so --sif would have every
    # condition measured against whichever index was built first.
    try:
        replay.check_conditions([("a", []), ("b", ["--sif", "--sif-a", "1e-4"])])
    except SystemExit as e:
        assert "--sif" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_search_only_flags_are_accepted():
    replay.check_conditions([("plain", []), ("mx", ["--maxsim", "--maxsim-pool", "48"])])


# --- the cluster bootstrap ---------------------------------------------------

def rows_for(spec):
    """spec: {instance: [(rank_a, rank_b), ...]}"""
    out = []
    for inst, pairs in spec.items():
        for ra, rb in pairs:
            out.append({"instance_id": inst, "ranks": {"a": ra, "b": rb}})
    return out


def test_clustered_and_naive_agree_when_every_instance_has_one_query():
    # With no clustering there is nothing to correct for, so the two
    # procedures are sampling the same thing.
    rows = rows_for({f"i{j}": [(1, 2)] for j in range(40)})
    _, lo_c, hi_c, n_cl = replay.bootstrap_ci(rows, "a", "b", clustered=True)
    _, lo_n, hi_n, _ = replay.bootstrap_ci(rows, "a", "b", clustered=False)
    assert n_cl == 40
    assert abs((hi_c - lo_c) - (hi_n - lo_n)) < 0.05


def test_clustering_widens_the_interval_when_queries_pile_into_one_instance():
    # This is the defect. One instance contributed 99 of 887 real queries.
    # Treating them as 99 independent observations reports a CI the data
    # cannot support.
    spec = {"big": [(1, 5)] * 60}
    spec.update({f"i{j}": [(5, 1)] for j in range(10)})
    rows = rows_for(spec)
    _, lo_c, hi_c, n_cl = replay.bootstrap_ci(rows, "a", "b", clustered=True)
    _, lo_n, hi_n, _ = replay.bootstrap_ci(rows, "a", "b", clustered=False)
    assert n_cl == 11
    assert (hi_c - lo_c) > (hi_n - lo_n) * 1.5


def test_the_point_estimate_is_unaffected_by_the_resampling_scheme():
    rows = rows_for({"big": [(1, 5)] * 30, "small": [(5, 1)]})
    pa, *_ = replay.bootstrap_ci(rows, "a", "b", clustered=True)
    pb, *_ = replay.bootstrap_ci(rows, "a", "b", clustered=False)
    assert abs(pa - pb) < 1e-12


def test_a_miss_scores_zero_reciprocal_rank():
    rows = rows_for({"i": [(None, 1)]})
    point, *_ = replay.bootstrap_ci(rows, "a", "b")
    assert point == -1.0


def test_bootstrap_on_no_rows_returns_zeros_rather_than_dividing_by_zero():
    assert replay.bootstrap_ci([], "a", "b") == (0.0, 0.0, 0.0, 0)


def test_the_bootstrap_is_deterministic():
    rows = rows_for({"i1": [(1, 3), (2, 4)], "i2": [(3, 1)]})
    assert replay.bootstrap_ci(rows, "a", "b") == replay.bootstrap_ci(rows, "a", "b")
