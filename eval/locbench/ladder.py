#!/usr/bin/env python3
"""Alternation-ladder decomposition for agentic guesses (RESEARCH.md §16.3).

An agent that doesn't know a name types its guess distribution as a regex
ladder: `writeParquet\\|save_parquet\\|to_parquet`. Two separate things live
in such a pattern and must not be conflated:

- the agent's INTENT: alternation over candidate spellings (the rungs);
- the ENGINE's semantics: ripgrep's regex engine (which `semgrep -e` and
  `rg` both use) treats `\\|` as a LITERAL pipe — a BRE-habit ladder is a
  dead search, matching a `|` character that occurs nowhere.

`parse()` recovers the intent and flags the mismatch. The translations are
what guessplay feeds ranked search: T1 joins the rung literals with casing
preserved (the engine's code-aware tokenizer already splits camel/snake);
T2 pre-splits as an over-translation control.
"""

import re

_METACHARS = set(".^$*+?()[]{}\\")
_IDENT = re.compile(
    r"^(def |class |fn |func |macro )?[A-Za-z_][A-Za-z0-9_]*([._:\-/][A-Za-z0-9_]+)*$")
_PHRASE = re.compile(r"""^[\w\s'"=/.,\-]+$""")
_CAMEL_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _split_top_level(pattern, escaped):
    """Split on `|` (escaped=False) or `\\|` (True) at bracket depth 0.
    Returns None when the pattern's groups are unbalanced — not decomposable."""
    parts, buf = [], []
    depth = 0
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if nxt == "|" and depth == 0 and escaped:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(c)
                buf.append(nxt)
            i += 2
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return None
        if c == "|" and depth == 0 and not escaped:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if depth != 0:
        return None
    parts.append("".join(buf))
    return parts


def _has_top_level(pattern, escaped):
    parts = _split_top_level(pattern, escaped)
    return parts is not None and len(parts) > 1


def _normalize_rung(raw):
    """Strip anchors, unescape, classify."""
    s = raw.strip()
    for anchor in ("\\b", "^"):
        while s.startswith(anchor):
            s = s[len(anchor):]
    for anchor in ("\\b", "$"):
        while s.endswith(anchor) and not s.endswith("\\" + anchor):
            s = s[: -len(anchor)]
    # Unescape: `\.` -> `.`, `\(` -> `(`, etc.
    literal = re.sub(r"\\(.)", r"\1", s)
    # Classification runs on the ESCAPE-AWARE form: a metachar that was
    # escaped in `s` is literal text, an unescaped one is regex machinery.
    unescaped_meta = bool(re.search(r"(?<!\\)[.^$*+?()\[\]{}]", s)) or \
        bool(re.search(r"\\[wsdWSD]", s))
    if not unescaped_meta and _IDENT.match(literal.strip()):
        kind = "identifier"
    elif not unescaped_meta and _PHRASE.match(literal):
        kind = "phrase"
    else:
        kind = "regex"
    return {"raw": raw, "literal": literal.strip(), "kind": kind}


def parse(patterns):
    """One invocation's pattern(s) -> a Ladder dict.

    `patterns` is a string or a list of strings (multiple `-e PAT` arguments
    form one ladder whose rungs span the patterns).
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    rungs = []
    sep = None
    mismatch = False
    decomposable = True
    for pat in patterns:
        if _has_top_level(pat, escaped=False):
            parts = _split_top_level(pat, escaped=False)
            sep = sep or "|"
        elif _has_top_level(pat, escaped=True):
            parts = _split_top_level(pat, escaped=True)
            sep = sep or "\\|"
            mismatch = True
        else:
            parts = _split_top_level(pat, escaped=False)
            if parts is None:
                decomposable = False
                parts = [pat]
        if parts is None:
            decomposable = False
            parts = [pat]
        rungs += [_normalize_rung(p) for p in parts if p.strip()]
    if not rungs:
        rungs = [_normalize_rung(patterns[0] if patterns else "")]
    return {
        "rungs": rungs,
        "n_rungs": len(rungs),
        "sep": sep,
        "decomposable": decomposable,
        "engine_semantics_mismatch": mismatch,
    }


def _strip_meta(text):
    return re.sub(r"[.^$*+?()\[\]{}\\]", " ", text)


def translate_t1(ladder):
    """Rung literals, deduped in order, space-joined; casing preserved.
    Regex rungs are dropped unless the whole ladder is regex (then a
    best-effort metachar strip)."""
    non_regex = [r for r in ladder["rungs"] if r["kind"] != "regex"]
    pool = non_regex if non_regex else ladder["rungs"]
    seen, parts = set(), []
    for r in pool:
        lit = r["literal"] if r["kind"] != "regex" else _strip_meta(r["literal"])
        for word in lit.split():
            if word.lower() not in seen:
                seen.add(word.lower())
                parts.append(word)
    return " ".join(parts)


def translate_t2(ladder):
    """T1, then camel/snake pre-split with dedupe — the over-translation
    control (the engine's own tokenizer already does this; doing it here
    destroys casing and whole-identifier signal on purpose)."""
    seen, parts = set(), []
    for word in translate_t1(ladder).split():
        for piece in word.split("_"):
            for m in _CAMEL_SPLIT.finditer(piece):
                w = m.group(0).lower()
                if len(w) >= 2 and w not in seen:
                    seen.add(w)
                    parts.append(w)
    return " ".join(parts)
