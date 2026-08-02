"""Ladder decomposition tests — every case is a verbatim pattern from the
shim logs (eval/data/locbench/runs), because the parser's job is the data
that exists, not the regexes one might imagine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "locbench"))

import ladder  # noqa: E402


def rungs(pat):
    return [r["literal"] for r in ladder.parse(pat)["rungs"]]


def test_bre_escaped_ladder_splits_and_flags_the_dead_search():
    l = ladder.parse(r"writeParquet\|save_parquet\|to_parquet")
    assert [r["literal"] for r in l["rungs"]] == [
        "writeParquet", "save_parquet", "to_parquet"]
    assert l["sep"] == "\\|"
    # ripgrep's engine reads \| as a literal pipe: this search was dead.
    assert l["engine_semantics_mismatch"]
    assert all(r["kind"] == "identifier" for r in l["rungs"])


def test_ere_ladder_splits_without_mismatch():
    l = ladder.parse("csrf|CSRF|X-CSRF|wtf")
    assert l["n_rungs"] == 4 and not l["engine_semantics_mismatch"]


def test_quoted_attribute_ladder():
    l = ladder.parse("""rel="canonical"|rel='canonical'|canonical_url""")
    assert l["n_rungs"] == 3
    assert l["rungs"][2]["kind"] == "identifier"


def test_def_prefixed_rungs_classify_as_identifiers():
    l = ladder.parse(r"def set_share_links\|macro set_share_links")
    assert l["n_rungs"] == 2
    assert all(r["kind"] == "identifier" for r in l["rungs"])


def test_single_guess_is_a_one_rung_ladder():
    l = ladder.parse("deepcopy")
    assert l["n_rungs"] == 1 and l["sep"] is None


def test_unbalanced_group_is_not_decomposable():
    l = ladder.parse(r"SOCKS(\(|socks")
    assert not l["decomposable"]


def test_eight_rung_ladder_from_the_logs():
    l = ladder.parse(
        "0.0.0.0|listenPort|listenAddress|SOCKS_SERVER|socks_addr|def start|def __init__|class SOCKS")
    assert l["n_rungs"] == 8


def test_escaped_dot_is_literal_text_not_regex():
    l = ladder.parse(r"requests\.|_requests\.|Session")
    assert l["n_rungs"] == 3
    assert l["rungs"][2]["literal"] == "Session"


def test_anchors_are_stripped_from_rungs():
    l = ladder.parse(r"^async def |^def ")
    assert [r["literal"] for r in l["rungs"]] == ["async def", "def"]


def test_multiple_e_args_form_one_ladder():
    l = ladder.parse(["writeParquet", "save_parquet"])
    assert l["n_rungs"] == 2


def test_t1_preserves_casing_and_dedupes():
    l = ladder.parse(r"writeParquet\|save_parquet\|to_parquet")
    assert ladder.translate_t1(l) == "writeParquet save_parquet to_parquet"


def test_t1_drops_regex_rungs_when_literals_exist():
    l = ladder.parse(r"requests\.|_requests\.|Session")
    t1 = ladder.translate_t1(l)
    assert "Session" in t1


def test_t2_pre_splits_camel_and_snake():
    l = ladder.parse(r"writeParquet\|save_parquet")
    assert ladder.translate_t2(l) == "write parquet save"


def test_character_class_pipe_is_not_a_ladder():
    l = ladder.parse("[a|b]c")
    assert l["n_rungs"] == 1
