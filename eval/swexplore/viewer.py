#!/usr/bin/env python3
"""Turn the §27 campaign into one self-contained HTML page.

    python3 eval/swexplore/viewer.py --run-id s27 --out /tmp/s27.html

Same idea as `eval/locbench/viewer.py`: the numbers and the trajectories behind
them on one page, nothing external, opens offline. The point of it is that a
table saying "+0.0018 [-0.0079,+0.0113]" is unreadable as *behaviour* — you
cannot tell from it whether the agents used the tool, what they typed, what came
back, or why it did not matter. This page shows the searches.

Statistics are computed here at build time with the same `ab_analyze.boot_ci`
and `mcnemar` the published numbers use, so the page cannot quietly disagree
with §27.3.

Trajectory detail is included for a *selected* subset (default 48) because
2,544 sessions of transcripts is 529 MB. Selection is deliberate and stated on
the page: the largest wins and losses on the primary, plus ties where the tool
was actually invoked, spread across languages. `--instances N` changes it.
"""

import argparse
import collections
import html
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
sys.path.insert(0, str(HERE.parent / "locbench"))
from ab_analyze import boot_ci, mcnemar  # noqa: E402

# Set from --arms/--contrasts; the defaults reproduce §27.
ARMS = ("cc", "cc-rg", "cc-sg")
ALL_LABEL = {
    "cc": "cc — built-in Grep only",
    "cc-rg": "cc-rg — Grep + ripgrep",
    "cc-sg": "cc-sg — Grep + semgrep",
    "sub-rg": "sub-rg — ripgrep only, no Grep",
    "sub-sg": "sub-sg — semgrep only, no Grep",
    "sub-sgb": "sub-sgb — semgrep + bridge expansion (§33)",
}
ALL_TOOLS = {
    "cc": "Read, Glob, Grep",
    "cc-rg": "Read, Glob, Grep, Bash(rg)",
    "cc-sg": "Read, Glob, Grep, Bash(sg)",
    "sub-rg": "Read, Glob, Bash(rg)",
    "sub-sg": "Read, Glob, Bash(sg)",
    "sub-sgb": "Read, Glob, Bash(sg --bridge-expand 8)",
}
ALL_ARM_TOOL = {"cc": None, "cc-rg": "rg", "cc-sg": "sg",
                "sub-rg": "rg", "sub-sg": "sg", "sub-sgb": "sg"}
ARM_LABEL = dict(ALL_LABEL)
ARM_TOOL = {"cc-rg": "rg", "cc-sg": "sg"}
CONTRASTS = (("cc-sg", "cc", "semgrep added vs Grep alone"),
             ("cc-rg", "cc", "ripgrep added vs Grep alone"),
             ("cc-sg", "cc-rg", "semgrep vs ripgrep, Bash held"))
# Plain-language gloss for every metric, so the page reads cold.
GLOSS = {
    "hitRegion@5": "Of the code regions a successful fix actually read, what "
                   "share did the agent's five answers overlap? <b>The primary "
                   "measure.</b>",
    "hitFile@5": "Same question at file level — did it reach the right files, "
                 "never mind the exact lines?",
    "ctxEff": "Of the lines the agent handed back, what share were actually "
              "useful? Rewards being relevant <i>and</i> compact.",
    "nDCG@500": "Coverage within a 500-line budget, rewarding useful regions "
                "ranked earlier.",
    "recall@100": "How much of the needed code is reachable in the first 100 "
                  "lines — the tight-budget view.",
    "precision": "Line-level precision of the returned regions.",
    "cost $": "US dollars of model spend for that one session.",
    "turns": "How many back-and-forth steps the agent took.",
}
ENDPOINTS = [("hit_region_rate", "hitRegion@5"), ("hit_file_rate", "hitFile@5"),
             ("context_efficiency", "ctxEff"), ("ndcg_at_500", "nDCG@500"),
             ("recall_at_100", "recall@100"), ("precision", "precision"),
             ("total_cost_usd", "cost $"), ("num_turns", "turns")]
MAX_OUT = 2600          # chars of a single search's output kept in the bundle
MAX_ISSUE = 4000


def val(r, k):
    if k in ("total_cost_usd", "num_turns"):
        return (r.get("agent") or {}).get(k) or 0
    return r["metrics"].get(k)


def load(run_id):
    by = {}
    for a in ARMS:
        p = DATA / "results" / f"{run_id}-{a}.jsonl"
        by[a] = {json.loads(l)["instance_id"]: json.loads(l)
                 for l in p.read_text().splitlines() if l.strip()}
    side = {json.loads(l)["instance_id"]: json.loads(l)
            for l in (DATA / "sidecar.jsonl").read_text().splitlines() if l.strip()}
    gold = {}
    for l in (DATA / "bench.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            gold[r["instance_id"]] = (r.get("ground_truth") or {})
    return by, side, gold


def searches_of(run_id, iid, arm):
    """Every non-blocked invocation with the bytes the agent actually saw."""
    d = DATA / "runs" / run_id / iid / arm
    log = d / "shim_log.jsonl"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("blocked") or e.get("tool") not in ("rg", "sg"):
            continue
        body = ""
        f = d / "searches" / (e.get("stdout_file") or "")
        if f.is_file():
            body = f.read_text(errors="replace")[:MAX_OUT]
        out.append({"argv": e.get("argv") or [], "exit": e.get("exit"),
                    "bytes": e.get("stdout_bytes") or 0,
                    "ms": e.get("wall_ms"), "out": body})
    return out


MAX_STEP_TEXT = 700
MAX_TOOL_IN = 260
MAX_TOOL_OUT = 420
MAX_STEPS = 60


def timeline_of(run_id, iid, arm):
    """The agent's actual step-by-step trajectory from the stream-json transcript.

    Searches alone do not show what the agent was *doing* — why it searched
    where it did, what it read afterwards, when it gave up on a thread. This
    walks the transcript in order and keeps three kinds of step: the model's
    own reasoning text, each tool call with its input, and the head of what
    that call returned. Truncated hard, because 2,544 transcripts is 529 MB and
    the page has to stay openable.
    """
    d = DATA / "runs" / run_id / iid / arm
    tf = d / "transcript.jsonl"
    if not tf.is_file():
        return []
    steps, pending = [], {}
    for line in tf.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            t = blk.get("type")
            if t == "text" and (blk.get("text") or "").strip():
                steps.append({"k": "say", "v": blk["text"].strip()[:MAX_STEP_TEXT]})
            elif t == "tool_use":
                inp = blk.get("input") or {}
                if isinstance(inp, dict):
                    # Show the field that actually says what was asked for.
                    key = next((x for x in ("command", "pattern", "file_path",
                                            "path", "query") if x in inp), None)
                    brief = str(inp.get(key)) if key else json.dumps(inp)
                    if key in ("pattern", "path") and "path" in inp and key != "path":
                        brief += f"   in {inp['path']}"
                else:
                    brief = str(inp)
                steps.append({"k": "tool", "name": blk.get("name") or "?",
                              "v": brief[:MAX_TOOL_IN], "id": blk.get("id")})
                pending[blk.get("id")] = len(steps) - 1
            elif t == "tool_result":
                c = blk.get("content")
                txt = c if isinstance(c, str) else json.dumps(c)
                ix = pending.get(blk.get("tool_use_id"))
                if ix is not None:
                    steps[ix]["out"] = (txt or "")[:MAX_TOOL_OUT]
    return steps[:MAX_STEPS]


def region_hits(pred, core):
    """Which gold regions each prediction overlaps — the page's hit/miss chips."""
    hits = []
    for c in core:
        got = any(p["path"] == c["path"] and p["start"] <= c["end"]
                  and c["start"] <= p["end"] for p in pred)
        hits.append(got)
    return hits


def pick(by, side, n):
    ids = sorted(set.intersection(*(set(by[a]) for a in ARMS)))
    K = "hit_region_rate"
    scored = []
    ta, tb = CONTRASTS[0][0], CONTRASTS[0][1]
    tool = ALL_ARM_TOOL.get(ta)
    for i in ids:
        d = val(by[ta][i], K) - val(by[tb][i], K)
        used = ((by[ta][i].get("agent") or {}).get("n_by_tool") or {}).get(tool, 0)
        scored.append((d, used, i))
    # Wins and losses are split into "sg actually ran" and "sg never ran".
    # Both belong on the page, and the second group is the point: the single
    # largest win in the whole campaign (+0.667) is an instance where no arm
    # invoked sg at all, which is what the null looks like from the inside.
    # Selecting only by |delta| would fill the browser with that case and hide
    # the sessions where the tool was genuinely exercised.
    q = n // 5
    wins_u = sorted([s for s in scored if s[0] > 0 and s[1] > 0], reverse=True)[:q]
    wins_n = sorted([s for s in scored if s[0] > 0 and s[1] == 0], reverse=True)[:q // 2]
    loss_u = sorted([s for s in scored if s[0] < 0 and s[1] > 0])[:q]
    loss_n = sorted([s for s in scored if s[0] < 0 and s[1] == 0])[:q // 2]
    got = wins_u + wins_n + loss_u + loss_n
    ties = sorted([s for s in scored if s[0] == 0 and s[1] > 0], key=lambda s: -s[1])
    return [i for _, _, i in got + ties[:max(0, n - len(got))]]


def _payload(obj) -> str:
    """JSON for embedding inside a <script> tag.

    `json.dumps` does not escape `<`, so a corpus string containing
    `</script>` closes the tag early and the ENTIRE data payload becomes
    unparseable — the page then renders its static header and nothing else,
    silently. s32 hit this twice (web repos carry HTML in their sources), and
    `<!--` is the same hazard by another route. Escaping `<` as \u003c is
    valid JSON, valid JS, and closes both.
    """
    return json.dumps(obj).replace("<", "\\u003c")


def esc(s):
    return html.escape(str(s), quote=True)


def build(run_id, n_detail, frame_n=0):
    by, side, gold = load(run_id)
    ids = sorted(set.intersection(*(set(by[a]) for a in ARMS)))
    if frame_n:
        # Pin to a ladder prefix. Without this a page built while a later rung
        # is still writing rows describes a frame that changes under it.
        order = json.loads((DATA / "ladder-frame.json").read_text())["order"][:frame_n]
        keep = set(order)
        ids = [i for i in ids if i in keep]
    detail_ids = [i for i in pick(by, side, n_detail) if i in set(ids)]

    # ---- headline stats, computed with the published convention -----------
    contrasts = []
    for a, b, title in CONTRASTS:
        rows = []
        for k, lbl in ENDPOINTS:
            pairs = [(val(by[a][i], k), val(by[b][i], k)) for i in ids]
            d, lo, hi = boot_ci(pairs)
            w, l, p = mcnemar(pairs)
            rows.append({"m": lbl, "d": d, "lo": lo, "hi": hi, "w": w, "l": l, "p": p,
                         "sig": not (lo <= 0 <= hi)})
        contrasts.append({"title": title, "rows": rows})

    means = {a: {lbl: st.mean(val(by[a][i], k) for i in ids)
                 for k, lbl in ENDPOINTS} for a in ARMS}

    # ---- ladder -----------------------------------------------------------
    frame = json.loads((DATA / "ladder-frame.json").read_text())["order"]
    pilot = set(json.loads((DATA / "pilot-frame.json").read_text())["instances"])
    # The rung decomposition only means something for the contrast whose rungs
    # these were. For any other arm set it is omitted rather than mislabelled.
    ladder = []
    if (CONTRASTS[0][0], CONTRASTS[0][1]) == ("cc-sg", "cc"):
        for lbl, s_ in (("R0 pilot", [i for i in ids if i in pilot]),
                        ("R1 independent", [i for i in frame[31:150] if i in ids]),
                        ("R2 new only", [i for i in frame[150:] if i in ids]),
                        ("pooled", ids)):
            pr = [(val(by["cc-sg"][i], "hit_region_rate"),
                   val(by["cc"][i], "hit_region_rate")) for i in s_]
            d, lo, hi = boot_ci(pr)
            ladder.append({"label": lbl, "n": len(s_), "d": d, "lo": lo, "hi": hi})

    inv = {a: sum(1 for i in ids
                  if ((by[a][i].get("agent") or {}).get("n_by_tool") or {}).get(t, 0) > 0)
           for a, t in ARM_TOOL.items()}

    # ---- how the two Bash tools were actually used ------------------------
    # The headline contrast is null, so the interesting comparison is not
    # "which found more" but "how differently they behave". Computed from the
    # shim logs, which record every real invocation.
    usage = {}
    for arm, tool in ARM_TOOL.items():
        calls, sess = [], 0
        for i in ids:
            s = searches_of(run_id, i, arm)
            if s:
                sess += 1
            calls += s
        pats = [next((x for x in c["argv"] if not x.startswith("-")), "") for c in calls]
        nbytes = [c["bytes"] for c in calls] or [0]
        usage[tool] = {
            "sessions": sess, "calls": len(calls),
            "per": len(calls) / max(sess, 1),
            "words": st.median([len(p.split()) for p in pats if p]) if pats else 0,
            "regex": sum(1 for p in pats if any(ch in p for ch in "|\\[](){}^$*+?")) / max(len(pats), 1),
            "med_bytes": st.median(nbytes), "mean_bytes": st.mean(nbytes),
            "empty": sum(1 for c in calls if c["bytes"] == 0) / max(len(calls), 1),
        }
    # Grep displacement: does the Bash tool replace the built-in, or add to it?
    grep = {a: st.mean((by[a][i].get("agent") or {}).get("grep_calls") or 0 for i in ids)
            for a in ARMS}
    bash = {a: st.mean((by[a][i].get("agent") or {}).get("bash_calls") or 0 for i in ids)
            for a in ARMS}

    langs = collections.Counter(side[i]["language"] for i in ids)
    perlang = []
    for lang, cnt in langs.most_common():
        s = [i for i in ids if side[i]["language"] == lang]
        if len(s) < 8:
            continue
        pr = [(val(by[CONTRASTS[0][0]][i], "hit_region_rate"),
               val(by[CONTRASTS[0][1]][i], "hit_region_rate")) for i in s]
        d, lo, hi = boot_ci(pr)
        perlang.append({"lang": lang, "n": len(s), "d": d, "lo": lo, "hi": hi})

    # ---- per-instance detail ----------------------------------------------
    detail = []
    for i in detail_ids:
        core = (gold[i].get("read_core_regions") or [])
        arms_d = {}
        for a in ARMS:
            r = by[a][i]
            pred = r.get("regions") or []
            arms_d[a] = {
                "regions": pred,
                "hits": region_hits(pred, core),
                "metrics": {lbl: val(r, k) for k, lbl in ENDPOINTS},
                "searches": searches_of(run_id, i, a),
                "steps": timeline_of(run_id, i, a),
                "cost": (r.get("agent") or {}).get("total_cost_usd") or 0,
                "turns": (r.get("agent") or {}).get("num_turns") or 0,
            }
        detail.append({
            "id": i, "lang": side[i]["language"], "repo": side[i]["repo"],
            "dataset": side[i]["dataset"],
            "issue": (side[i]["problem_statement"] or "")[:MAX_ISSUE],
            "core": core, "arms": arms_d,
            "delta": (arms_d[CONTRASTS[0][0]]["metrics"]["hitRegion@5"]
                      - arms_d[CONTRASTS[0][1]]["metrics"]["hitRegion@5"]),
        })

    total_cost = sum((by[a][i].get("agent") or {}).get("total_cost_usd") or 0
                     for a in ARMS for i in ids)
    return {"run": run_id, "n": len(ids), "arms": ARMS, "arm_label": ARM_LABEL,
            "means": means, "contrasts": contrasts, "ladder": ladder,
            "invocation": {a: {"used": inv[a], "n": len(ids)} for a in ARM_TOOL},
            "perlang": perlang, "detail": detail, "cost": total_cost,
            "usage": usage, "grep": grep, "bash": bash,
            "arm_tool": {a: ALL_ARM_TOOL.get(a) for a in ARMS},
            "primary_arm": CONTRASTS[0][0], "primary_tool": ALL_ARM_TOOL.get(CONTRASTS[0][0]) or "",
            "n_detail": len(detail_ids)}


# --------------------------------------------------------------------------- page
CSS = """
:root{
  --bg:#fbfaf8; --panel:#fff; --ink:#1a1916; --ink2:#585349; --ink3:#8c8578;
  --rule:#e6e1d8; --accent:#7c4a2d; --accent-soft:#f3e9e2;
  --good:#2f6b4f; --bad:#9c3b2e; --flat:#8c8578; --code:#f6f3ee;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14130f; --panel:#1c1a16; --ink:#f0ece4; --ink2:#b8b1a4; --ink3:#847d70;
  --rule:#2e2b24; --accent:#d99a72; --accent-soft:#2a211a;
  --good:#6fbf95; --bad:#e0836f; --flat:#847d70; --code:#221f1a;}}
:root[data-theme="dark"]{
  --bg:#14130f; --panel:#1c1a16; --ink:#f0ece4; --ink2:#b8b1a4; --ink3:#847d70;
  --rule:#2e2b24; --accent:#d99a72; --accent-soft:#2a211a;
  --good:#6fbf95; --bad:#e0836f; --flat:#847d70; --code:#221f1a;}
:root[data-theme="light"]{
  --bg:#fbfaf8; --panel:#fff; --ink:#1a1916; --ink2:#585349; --ink3:#8c8578;
  --rule:#e6e1d8; --accent:#7c4a2d; --accent-soft:#f3e9e2;
  --good:#2f6b4f; --bad:#9c3b2e; --flat:#8c8578; --code:#f6f3ee;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:29px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em;text-wrap:balance}
h2{font-size:19px;margin:44px 0 12px;letter-spacing:-.01em;
   padding-bottom:7px;border-bottom:1px solid var(--rule)}
h3{font-size:14px;margin:22px 0 8px;color:var(--ink2);
   text-transform:uppercase;letter-spacing:.09em;font-weight:600}
p{margin:0 0 12px;max-width:76ch;color:var(--ink2)}
.sub{color:var(--ink3);font-size:13.5px;margin-bottom:26px}
.hero{background:var(--panel);border:1px solid var(--rule);border-radius:10px;
  padding:22px 24px;margin:22px 0 8px}
.hero .big{font:600 34px/1.1 var(--mono);letter-spacing:-.02em}
.hero .ci{font:13px/1.5 var(--mono);color:var(--ink3);margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:18px 0}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:9px;padding:13px 15px}
.kpi .v{font:600 21px/1.2 var(--mono)}
.kpi .k{font-size:11.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.07em;margin-top:3px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:560px}
th,td{text-align:right;padding:7px 11px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink3);font-weight:600}
td.m{font-family:var(--mono)}
.pos{color:var(--good)} .neg{color:var(--bad)} .flat{color:var(--flat)}
.sig::after{content:"*";color:var(--accent);font-weight:700}
.bar{position:relative;height:9px;background:var(--rule);border-radius:5px;min-width:130px}
.bar i{position:absolute;top:0;bottom:0;border-radius:5px;background:var(--accent)}
.bar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.55}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;
  border:1px solid var(--rule);color:var(--ink2);background:var(--panel)}
.pill.hit{color:var(--good);border-color:var(--good)}
.pill.miss{color:var(--bad);border-color:var(--bad)}
.pill.tool{background:var(--accent-soft);border-color:transparent;color:var(--accent);font-family:var(--mono)}
/* A zero-invocation session is the single most important state on this page:
   the treatment was never delivered, so any delta on that row is noise. It
   gets the muted, struck treatment rather than looking like a small number. */
.pill.tool.zero{background:transparent;border:1px dashed var(--rule);color:var(--ink3)}
.arm.unused{opacity:.72}
.arm.unused h4{color:var(--ink3)}
.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:14px 0 18px}
select,input,button{font:13px/1.4 inherit;padding:6px 10px;border-radius:7px;
  border:1px solid var(--rule);background:var(--panel);color:var(--ink)}
button{cursor:pointer}
.inst{background:var(--panel);border:1px solid var(--rule);border-radius:10px;
  margin-bottom:12px;overflow:hidden}
.ihead{display:flex;gap:12px;align-items:center;padding:12px 16px;cursor:pointer}
.ihead:hover{background:var(--accent-soft)}
.ihead .id{font-family:var(--mono);font-size:12.5px;flex:1;min-width:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ibody{display:none;padding:0 16px 18px;border-top:1px solid var(--rule)}
.inst.open .ibody{display:block}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;margin-top:12px}
.arm{border:1px solid var(--rule);border-radius:9px;padding:12px 13px;background:var(--bg)}
.arm h4{margin:0 0 8px;font-size:12.5px;font-family:var(--mono);color:var(--accent)}
pre{background:var(--code);border:1px solid var(--rule);border-radius:7px;
  padding:9px 11px;overflow-x:auto;font:11.5px/1.55 var(--mono);margin:5px 0 0;
  white-space:pre;max-height:250px}
.q{font-family:var(--mono);font-size:12px;background:var(--accent-soft);
  color:var(--accent);padding:4px 8px;border-radius:6px;display:inline-block;
  margin-top:9px;word-break:break-word;white-space:normal}
.issue{background:var(--code);border:1px solid var(--rule);border-radius:7px;
  padding:11px 13px;font-size:12.5px;max-height:210px;overflow:auto;
  white-space:pre-wrap;color:var(--ink2);margin-top:6px}
.small{font-size:12px;color:var(--ink3)}
.none{color:var(--ink3);font-style:italic;font-size:12.5px}
.traj{margin-top:12px;border-top:1px dashed var(--rule);padding-top:10px}
.traj summary{cursor:pointer;font-size:12px;color:var(--accent);
  text-transform:uppercase;letter-spacing:.07em}
.step{display:flex;gap:8px;margin:7px 0;font-size:12px;line-height:1.5}
.step .n{color:var(--ink3);font-family:var(--mono);font-size:10.5px;min-width:20px;
  text-align:right;padding-top:2px}
.step .b{flex:1;min-width:0}
.step.say .b{color:var(--ink2)}
.tname{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:0 6px;
  border-radius:99px;background:var(--accent-soft);color:var(--accent);margin-right:6px}
.tname.read{background:transparent;border:1px solid var(--rule);color:var(--ink3)}
.targ{font-family:var(--mono);font-size:11.5px;word-break:break-word}
.tout{margin-top:3px;color:var(--ink3);font-family:var(--mono);font-size:11px;
  white-space:pre-wrap;max-height:76px;overflow:hidden;
  -webkit-mask-image:linear-gradient(#000 60%,transparent)}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 13px;margin:14px 0;
  color:var(--ink2);font-size:13.5px;max-width:76ch}
"""

JS = """
const D = window.__DATA__;
const f = (x,n=4)=> (x>=0?'+':'') + x.toFixed(n);
const cls = (d,sig)=> (sig? (d>0?'pos':'neg') : 'flat') + (sig?' sig':'');

function armCard(inst, arm){
  const a = inst.arms[arm], t = D.arm_tool[arm] || '';
  const unused = t && a.searches.length===0;
  let s = `<div class="arm${unused?' unused':''}"><h4>${D.arm_label[arm]}</h4>`;
  s += `<div class="small">hitRegion ${a.metrics['hitRegion@5'].toFixed(2)}
        · $${a.cost.toFixed(3)} · ${a.turns} turns</div>`;
  if(a.searches.length){
    for(const q of a.searches){
      s += `<div class="q">${t} ${q.argv.map(x=>JSON.stringify(x)).join(' ')}</div>`;
      s += `<pre>${q.out ? esc(q.out) : '(no output, exit '+q.exit+')'}</pre>`;
    }
  } else {
    s += `<div class="none">never invoked ${t||'a Bash search tool'} — used the built-in Grep tool</div>`;
  }
  if(a.steps && a.steps.length){
    s += `<details class="traj"><summary>▸ full trajectory (${a.steps.length} steps)</summary>`;
    a.steps.forEach((st,ix)=>{
      if(st.k==='say'){
        s += `<div class="step say"><span class="n">${ix+1}</span><span class="b">${esc(st.v)}</span></div>`;
      } else {
        const cls = (st.name==='Read'||st.name==='Glob') ? 'tname read' : 'tname';
        s += `<div class="step"><span class="n">${ix+1}</span><span class="b">`
           + `<span class="${cls}">${esc(st.name)}</span>`
           + `<span class="targ">${esc(st.v)}</span>`
           + (st.out ? `<div class="tout">${esc(st.out)}</div>` : '')
           + `</span></div>`;
      }
    });
    s += `</details>`;
  }
  s += `<div class="small" style="margin-top:10px">answered:</div>`;
  s += a.regions.length
    ? a.regions.map(r=>`<div class="small" style="font-family:var(--mono)">${esc(r.path)}:${r.start}-${r.end}</div>`).join('')
    : `<div class="none">no regions parsed</div>`;
  return s + `</div>`;
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

function render(){
  const langSel = document.getElementById('lang').value;
  const kind = document.getElementById('kind').value;
  const box = document.getElementById('list');
  box.innerHTML='';
  let shown=0;
  for(const inst of D.detail){
    if(langSel!=='all' && inst.lang!==langSel) continue;
    if(kind==='win' && !(inst.delta>0)) continue;
    if(kind==='loss' && !(inst.delta<0)) continue;
    if(kind==='tie' && inst.delta!==0) continue;
    shown++;
    const d=inst.delta, dc = d>0?'pos':(d<0?'neg':'flat');
    const el=document.createElement('div'); el.className='inst';
    el.innerHTML = `<div class="ihead">
        <span class="pill">${inst.lang}</span>
        <span class="id">${esc(inst.id)}</span>
        <span class="pill tool${inst.arms[D.primary_arm].searches.length?'':' zero'}">${D.primary_tool} ${inst.arms[D.primary_arm].searches.length}×</span>
        <span class="m ${dc}" style="font-family:var(--mono);min-width:74px;text-align:right">${f(d,3)}</span>
      </div>
      <div class="ibody">
        <h3>The issue the agent was given</h3>
        <div class="issue">${esc(inst.issue)}</div>
        <h3>Ground truth — regions successful repair trajectories read</h3>
        <div>${inst.core.map((c,ix)=>{
            const hit = inst.arms[D.primary_arm].hits[ix];
            return `<span class="pill ${hit?'hit':'miss'}" style="margin:2px 4px 2px 0;font-family:var(--mono)">
              ${esc(c.path)}:${c.start}-${c.end} ${hit?'✓ '+D.primary_arm:'✗ '+D.primary_arm}</span>`;}).join('')}</div>
        <h3>What each arm searched, and what came back</h3>
        <div class="grid3">${D.arms.map(a=>armCard(inst,a)).join('')}</div>
      </div>`;
    el.querySelector('.ihead').onclick=()=>el.classList.toggle('open');
    box.appendChild(el);
  }
  document.getElementById('count').textContent =
    `${shown} of ${D.detail.length} shown (trajectories captured for ${D.n_detail} of ${D.n} instances)`;
}
document.getElementById('lang').onchange=render;
document.getElementById('kind').onchange=render;
document.getElementById('theme').onclick=()=>{
  const r=document.documentElement;
  const cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme', cur==='dark'?'light':'dark');
};
render();
"""


def render(b):
    prim = b["contrasts"][0]
    prow = prim["rows"][0]
    langs = sorted({d["lang"] for d in b["detail"]})
    u = b["usage"]
    arms = b["arms"]

    def sgn(x, n=4, pct=False):
        return f"{x:+.1%}" if pct else f"{x:+.{n}f}"

    def ctable(c):
        rows = ""
        for r in c["rows"]:
            klass = ("pos" if r["d"] > 0 else "neg") if r["sig"] else "flat"
            rows += (f'<tr><td>{esc(r["m"])}</td>'
                     f'<td class="m {klass}{" sig" if r["sig"] else ""}">{r["d"]:+.4f}</td>'
                     f'<td class="m flat">[{r["lo"]:+.4f}, {r["hi"]:+.4f}]</td>'
                     f'<td class="m">{r["w"]}/{r["l"]}</td>'
                     f'<td class="m flat">{r["p"]:.3f}</td></tr>')
        return (f'<h3>{esc(c["title"])}</h3><div class="scroll"><table>'
                f'<tr><th>measure</th><th>difference</th><th>95% confidence</th>'
                f'<th>wins/losses</th><th>sign p</th></tr>{rows}</table></div>')

    armrows = "".join(
        f'<tr><td class="m">{esc(a)}</td><td class="m">{esc(ALL_TOOLS.get(a, "?"))}</td>'
        f'<td style="text-align:left">{esc(ARM_LABEL.get(a, ALL_LABEL.get(a, a)).split("—")[-1].strip())}</td></tr>'
        for a in arms)

    gloss = "".join(
        f'<tr><td class="m">{esc(lbl)}</td><td style="text-align:left">{GLOSS.get(lbl, "")}</td></tr>'
        for _, lbl in ENDPOINTS)

    mrows = "".join(
        f'<tr><td>{esc(lbl)}</td>' +
        "".join(f'<td class="m">{b["means"][a][lbl]:.4f}</td>' for a in arms) + '</tr>'
        for _, lbl in ENDPOINTS)

    usagerows = ""
    if u:
        keys = sorted(u)
        def row(label, fmt, k):
            return (f'<tr><td>{label}</td>' +
                    "".join(f'<td class="m">{fmt(u[t][k])}</td>' for t in keys) + '</tr>')
        usagerows = (
            f'<div class="scroll"><table><tr><th>behaviour</th>'
            + "".join(f'<th>{esc(t)}</th>' for t in keys) + '</tr>'
            + row("sessions that used it", lambda v: f"{v}", "sessions")
            + row("calls per session", lambda v: f"{v:.1f}", "per")
            + row("median query length", lambda v: f"{v:.0f} words", "words")
            + row("queries using regex syntax", lambda v: f"{v:.0%}", "regex")
            + row("median bytes returned", lambda v: f"{v:,.0f}", "med_bytes")
            + row("<b>mean</b> bytes returned", lambda v: f"{v:,.0f}", "mean_bytes")
            + row("searches returning nothing", lambda v: f"{v:.0%}", "empty")
            + '</table></div>')

    inv = " · ".join(f'<b>{a}</b> {v["used"]}/{v["n"]} ({v["used"]/v["n"]:.0%})'
                     for a, v in b["invocation"].items())

    greprows = "".join(
        f'<tr><td class="m">{esc(a)}</td><td class="m">{b["grep"][a]:.2f}</td>'
        f'<td class="m">{b["bash"][a]:.2f}</td>'
        f'<td class="m">{b["grep"][a]+b["bash"][a]:.2f}</td></tr>' for a in arms)

    pl = "".join(
        f'<tr><td>{esc(x["lang"])}</td><td class="m">{x["n"]}</td>'
        f'<td class="m {"pos" if x["lo"]>0 else "neg" if x["hi"]<0 else "flat"}">{x["d"]:+.4f}</td>'
        f'<td class="m flat">[{x["lo"]:+.4f}, {x["hi"]:+.4f}]</td></tr>' for x in b["perlang"])

    ladder_html = ""
    if b["ladder"]:
        lad = "".join(
            f'<tr><td>{esc(x["label"])}</td><td class="m">{x["n"]}</td>'
            f'<td class="m {"pos" if x["lo"]>0 else "flat"}">{x["d"]:+.4f}</td>'
            f'<td class="m flat">[{x["lo"]:+.4f}, {x["hi"]:+.4f}]</td></tr>'
            for x in b["ladder"])
        ladder_html = f"""<h2>How the estimate changed as the sample grew</h2>
<p>The same comparison at each stage. An early estimate that sits at the edge of what a
small sample can detect is the classic setup for shrinking toward nothing as data
arrives — which is what happened.</p>
<div class="scroll"><table><tr><th>stage</th><th>n</th><th>difference</th>
<th>95% confidence</th></tr>{lad}</table></div>"""

    verdict = ("no measurable difference" if not prow["sig"]
               else ("better" if prow["d"] > 0 else "worse"))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(b["title"])}</title><style>{CSS}</style></head><body>
<div class="wrap">
<button id="theme" style="float:right">◐ theme</button>
<h1>{esc(b["title"])}</h1>
<div class="sub">SWE-Explore · {b["n"]} tasks · {len(arms)} arms · {b["n"]*len(arms):,} agent
sessions · ${b["cost"]:.2f} · run <code>{esc(b["run"])}</code></div>

<h2>What this is</h2>
<p><b>The question.</b> A coding agent given a bug report has to find the code that needs
changing. It does that by searching the repository. This measures whether the search tool it
is given changes how well it finds that code — and what the tool costs.</p>
<p><b>The benchmark.</b> SWE-Explore is 848 real GitHub issues across 203 repositories and 10
languages. For each one it records the <i>code regions that agents who actually fixed the bug
read along the way</i>, intersected across several successful attempts. That is the answer key.
The agent under test returns at most five regions — a file path and a line range — and is scored
on how well those five overlap the key.</p>
<p><b>The arms.</b> Each arm is the same agent on the same task with the same prompt. The only
difference is which search tools exist:</p>
<div class="scroll"><table><tr><th>arm</th><th>tools available</th><th>meaning</th></tr>
{armrows}</table></div>
<div class="note"><b>Why an arm that differs by one flag.</b> Both arms are the same agent, same prompt, same tool description, same index — the only difference is whether the search tool expands the query with vocabulary mined from the repository's own bridge files. A contrast that narrow is what lets a difference be attributed to the mechanism rather than to the tool being present at all.</div>

<h2>How to read the numbers</h2>
<div class="scroll"><table><tr><th>measure</th><th>what it asks</th></tr>{gloss}</table></div>
<div class="note"><b>Difference, confidence, wins/losses.</b> Every comparison is
<i>paired</i>: the same task run under both arms, so task difficulty cancels. The
<b>difference</b> is the average change. The <b>95% confidence</b> interval is the range the
true value plausibly sits in — <b>if it contains zero, there is no detectable effect</b>. The
<b>wins/losses</b> count how many individual tasks moved each way, which catches an average
dragged around by a handful of tasks rather than a consistent shift.</div>

<div class="hero">
  <div class="small">HEADLINE — {esc(prim["title"])}, on {esc(ENDPOINTS[0][1])}</div>
  <div class="big {"pos" if prow["sig"] and prow["d"]>0 else "neg" if prow["sig"] and prow["d"]<0 else "flat"}">{prow["d"]:+.4f}</div>
  <div class="ci">95% confidence [{prow["lo"]:+.4f}, {prow["hi"]:+.4f}] · {prow["w"]} wins /
  {prow["l"]} losses · sign p={prow["p"]:.3f} — <b>{verdict}</b></div>
</div>

<h2>Average per arm</h2>
<div class="scroll"><table><tr><th>measure</th>{"".join(f"<th>{esc(a)}</th>" for a in arms)}</tr>
{mrows}</table></div>

<h2>Every comparison</h2>
{"".join(ctable(c) for c in b["contrasts"])}

<h2>Did the agents actually use the tool?</h2>
<p>A tool that is never called cannot change anything, so this is the first thing to check.</p>
<div class="note">{inv}</div>
{usagerows}
<h3>Built-in search vs the shell tool</h3>
<div class="scroll"><table><tr><th>arm</th><th>built-in Grep calls</th>
<th>shell calls</th><th>total searches</th></tr>{greprows}</table></div>

{f'<h2>By language</h2><p>Groups smaller than 8 tasks are omitted rather than reported.</p><div class="scroll"><table><tr><th>language</th><th>n</th><th>difference</th><th>95% confidence</th></tr>{pl}</table></div>' if pl else ''}

{ladder_html}

<h2>What the agents actually did</h2>
<p>Numbers hide behaviour. These are {b["n_detail"]} real sessions — click any row to see the
bug report, the answer key, and, for each arm, <b>every search it ran and what came back</b>.</p>
<div class="note"><b>What to look for.</b> A row marked <span class="pill tool zero">0×</span>
means that arm never invoked its search tool, so any difference on that row is run-to-run
randomness rather than the tool. Re-running an identical arm twice moves the score by about as
much as the difference being measured, which is what makes a small sample untrustworthy here.</div>
<div class="controls">
  <select id="lang"><option value="all">all languages</option>
  {"".join(f'<option>{esc(l)}</option>' for l in langs)}</select>
  <select id="kind"><option value="all">all outcomes</option>
    <option value="win">first arm won</option><option value="loss">first arm lost</option>
    <option value="tie">tied</option></select>
  <span class="small" id="count"></span>
</div>
<div id="list"></div>
</div>
<script>window.__DATA__={_payload(b)};</script><script>{JS}</script>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s27")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--contrasts", default="",
                    help="colon-paired, comma separated; first is the headline")
    ap.add_argument("--frame", type=int, default=0,
                    help="pin to the first N of the ladder order (a rung)")
    ap.add_argument("--title", default="")
    ap.add_argument("--grep-unblocked", action="store_true",
                    help="§32 regime: shell grep was available to BOTH arms, so "
                         "the sub-* labels' 'no Grep' is false — relabel them. A "
                         "viewer that mislabels the treatment is worse than none.")
    ap.add_argument("--instances", type=int, default=48)
    ap.add_argument("--out", type=Path, default=Path("/tmp/swexplore-viewer.html"))
    a = ap.parse_args()

    ARMS = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    ARM_TOOL = {x: ALL_ARM_TOOL[x] for x in ARMS if ALL_ARM_TOOL.get(x)}
    if a.grep_unblocked:
        # §32 unblocked shell grep in BOTH arms, so the sub-* labels' "no Grep"
        # is false. A viewer that mislabels the treatment is worse than none.
        # ARM_LABEL feeds the meaning column (viewer bug found 2026-08-15: it
        # read the un-relabelled ALL_LABEL, so a page built with grep unblocked
        # still told the reader "no Grep" beside a tools column listing grep).
        ARM_LABEL["sub-rg"] = "sub-rg — ripgrep + shell grep (two lexical tools)"
        ARM_LABEL["sub-sg"] = "sub-sg — semgrep + shell grep (semantic + lexical)"
        ARM_LABEL["sub-sgb"] = ("sub-sgb — the same, plus bridge expansion "
                                "(§33's one-flag treatment)")
        ALL_TOOLS["sub-rg"] = "Read, Glob, Bash(rg, grep)"
        ALL_TOOLS["sub-sg"] = "Read, Glob, Bash(sg, grep)"
        ALL_TOOLS["sub-sgb"] = "Read, Glob, Bash(sg --bridge-expand 8, grep)"
    if a.contrasts:
        CONTRASTS = tuple((p.split(":")[0].strip(), p.split(":")[1].strip(),
                           f"{p.split(':')[0].strip()} vs {p.split(':')[1].strip()}")
                          for p in a.contrasts.split(","))
    bundle = build(a.run_id, a.instances, a.frame)
    bundle["title"] = a.title or "SWE-Explore: which search tool should an agent have?"
    a.out.write_text(render(bundle))
    mb = a.out.stat().st_size / 1e6
    print(f"wrote {a.out} ({mb:.1f} MB) — {bundle['n']} tasks, "
          f"{bundle['n_detail']} with trajectories, arms={list(ARMS)}")
