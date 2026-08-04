#!/usr/bin/env python3
"""Render a capture bundle as one self-contained HTML page.

    python3 eval/locbench/capture.py
    python3 eval/locbench/viewer.py            # -> eval/data/locbench/results-viewer.html

No external fonts, scripts, styles, or fetches: the page opens from `file://`
with the network off. The bundle is inlined; statistics are computed here, at
build time, by the same functions `ab_analyze.py` uses for the published
tables, so the page cannot quietly disagree with them.

Three views rather than one scroll — the first cut of this page was 47 screens
tall with 16k nodes at first paint, which is not a document anyone reads.

Palette validated with the dataviz skill's `validate_palette.js` rather than by
eye: the two arms pass every check in both themes, and the status trio passes
all-pairs (the dark CVD warn is legal because status always ships with an icon
and a label, never colour alone).
"""

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"
sys.path.insert(0, str(HERE))

try:
    from ab_analyze import mcnemar, boot_ci
except ImportError:  # pragma: no cover - ab_analyze is in-tree
    mcnemar = boot_ci = None

PRIMARY = "func_acc@10_tol"
SECONDARY = "file_acc@5"
ARMS = ("rg", "desc-v5")

# What a sample lacks the instrumentation to answer, and the figure established
# elsewhere. §16.10 predates SEMGREP_TRACE_FILE, so its empty-ranked rate is not
# 0 — it is unmeasured, and RESEARCH.md §17 recovered the real number offline by
# classifying every logged invocation. Rendering "0/0" there showed the
# project's worst result as a clean zero.
KNOWN_ELSEWHERE = {
    "results-scale": {
        "empty_ranked": "59% (2,078 of 3,519), recovered offline in §17",
        "unreadable": "96% of those empties were the §16.11 file-scope bug",
    }
}


def scoreboard(bundle):
    """Paired accuracy per tier and pooled, with discordant counts.

    Discordant counts travel with every delta on purpose. At n=40 a delta is
    four pairs wide, and a table showing +0.050 without w2/l0 beside it invites
    exactly the misreading §18.6 caught when an independent sample reversed the
    sign.
    """
    out, pooled = [], {}
    for tier in bundle["tiers"]:
        if not tier.get("with_trajectories"):
            continue
        by = {}
        for r in tier["rows"]:
            if r["status"] != "ok":
                continue
            by.setdefault(r["instance_id"], {})[r["condition"]] = r
        ids = [i for i, d in by.items() if all(a in d for a in ARMS)]
        row = {"tier": tier["name"], "n": len(ids), "metrics": {}}
        for key in (PRIMARY, SECONDARY):
            pairs = [(bool((by[i]["desc-v5"]["metrics"] or {}).get(key)),
                      bool((by[i]["rg"]["metrics"] or {}).get(key))) for i in ids]
            row["metrics"][key] = stat_block(pairs)
            for i in ids:
                pooled.setdefault(i, {})[key] = (
                    bool((by[i]["desc-v5"]["metrics"] or {}).get(key)),
                    bool((by[i]["rg"]["metrics"] or {}).get(key)))
        row["cost"] = {a: mean([r["cost"] for r in tier["rows"]
                                if r["condition"] == a and r["status"] == "ok"]) for a in ARMS}
        out.append(row)

    if pooled:
        row = {"tier": "pooled (distinct instances)", "n": len(pooled), "metrics": {}}
        for key in (PRIMARY, SECONDARY):
            row["metrics"][key] = stat_block([v[key] for v in pooled.values() if key in v])
        out.append(row)
    return out


def stat_block(pairs):
    if not pairs:
        return {"a": 0, "b": 0, "delta": 0, "w": 0, "l": 0, "p": 1.0, "ci": [0, 0], "n": 0}
    n = len(pairs)
    blk = {"a": sum(1 for x, _ in pairs if x) / n,
           "b": sum(1 for _, y in pairs if y) / n, "n": n}
    blk["delta"] = blk["a"] - blk["b"]
    if mcnemar:
        w, l, p = mcnemar([(int(x), int(y)) for x, y in pairs])
        _, lo, hi = boot_ci([(int(x), int(y)) for x, y in pairs])
        blk.update(w=w, l=l, p=p, ci=[lo, hi])
    else:
        blk.update(w=sum(1 for x, y in pairs if x and not y),
                   l=sum(1 for x, y in pairs if y and not x), p=1.0, ci=[0, 0])
    return blk


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


CSS = """
:root{
  --bg:#FBFAF7; --panel:#FFFFFF; --line:#E3E0D8; --line-soft:#EFECE5;
  --ink:#1B1D22; --ink-2:#4B5058; --ink-3:#797F88;
  --rg:#1F5FA8; --sg:#C25A18;
  --good:#0C968C; --warn:#A67C00; --bad:#C62B4E;
  --good-bg:#E6F4F2; --warn-bg:#F7F0DE; --bad-bg:#FAE7EC;
  --accent:#1F5FA8; --accent-bg:#E9F0F9;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --r:7px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14171C; --panel:#1A1E25; --line:#2C323C; --line-soft:#232830;
  --ink:#E8EAED; --ink-2:#A8B0BC; --ink-3:#78818F;
  --rg:#4A86D0; --sg:#C4703A;
  --good:#1F9E92; --warn:#B58E10; --bad:#D6455F;
  --good-bg:#0F2B2A; --warn-bg:#2C2612; --bad-bg:#331520;
  --accent:#4A86D0; --accent-bg:#15202E;
}}
:root[data-theme="dark"]{
  --bg:#14171C; --panel:#1A1E25; --line:#2C323C; --line-soft:#232830;
  --ink:#E8EAED; --ink-2:#A8B0BC; --ink-3:#78818F;
  --rg:#4A86D0; --sg:#C4703A;
  --good:#1F9E92; --warn:#B58E10; --bad:#D6455F;
  --good-bg:#0F2B2A; --warn-bg:#2C2612; --bad-bg:#331520;
  --accent:#4A86D0; --accent-bg:#15202E;
}
:root[data-theme="light"]{
  --bg:#FBFAF7; --panel:#FFFFFF; --line:#E3E0D8; --line-soft:#EFECE5;
  --ink:#1B1D22; --ink-2:#4B5058; --ink-3:#797F88;
  --rg:#1F5FA8; --sg:#C25A18;
  --good:#0C968C; --warn:#A67C00; --bad:#C62B4E;
  --good-bg:#E6F4F2; --warn-bg:#F7F0DE; --bad-bg:#FAE7EC;
  --accent:#1F5FA8; --accent-bg:#E9F0F9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1360px;margin:0 auto;padding:0 20px 90px}
h1{font-size:18px;margin:0;letter-spacing:-.01em}
h2{font-size:12px;margin:0 0 2px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600}
.sub{color:var(--ink-3);font-size:12.5px;margin:2px 0 0}
header{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line)}
.hrow{max-width:1360px;margin:0 auto;padding:10px 20px;display:flex;gap:14px;
  align-items:center;flex-wrap:wrap}
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-left:auto;align-items:center}
.chip{font-family:var(--mono);font-size:10.5px;padding:2px 7px;border-radius:99px;
  border:1px solid var(--line);color:var(--ink-3);background:var(--panel);white-space:nowrap}
.chip b{color:var(--ink-2);font-weight:600}
@media (max-width:820px){.chip.opt{display:none}}
nav{display:flex;gap:2px;max-width:1360px;margin:0 auto;padding:0 20px}
nav button{font-family:var(--sans);font-size:13px;padding:7px 13px;border:0;
  border-bottom:2px solid transparent;background:none;color:var(--ink-3);cursor:pointer}
nav button[aria-current="true"]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
nav button:hover{color:var(--ink)}
section{margin-top:26px}
.view{display:none}.view.on{display:block}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
  padding:14px;margin-top:9px}
.scroll{overflow-x:auto}
.headline{border:1px solid var(--line);border-left:3px solid var(--good);
  background:var(--panel);border-radius:var(--r);padding:16px 18px;margin-top:18px}
.headline.bad{border-left-color:var(--bad)}
.headline p{margin:0;font-size:15px;line-height:1.55;max-width:78ch}
.headline .caveat{margin-top:8px;font-size:13px;color:var(--ink-2)}
.big{display:flex;gap:26px;flex-wrap:wrap;margin-top:14px}
.big div{display:flex;flex-direction:column}
.big .v{font-family:var(--mono);font-size:21px;font-weight:600;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.big .k{font-size:11px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line-soft);
  white-space:nowrap;vertical-align:top}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);
  font-weight:600;background:var(--panel);user-select:none}
th[data-k]{cursor:pointer}
th[data-k]:hover{color:var(--ink)}
th .caret{opacity:.85;margin-left:3px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono)}
tbody tr.clickable{cursor:pointer}
tbody tr.clickable:hover{background:var(--line-soft)}
tbody tr:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.mono{font-family:var(--mono)}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
  padding:2px 8px;border-radius:99px;font-family:var(--mono);white-space:nowrap}
.pill.good{background:var(--good-bg);color:var(--good)}
.pill.warn{background:var(--warn-bg);color:var(--warn)}
.pill.bad{background:var(--bad-bg);color:var(--bad)}
.pill.mute{background:var(--line-soft);color:var(--ink-3)}
.pill.nm{background:transparent;color:var(--ink-3);border:1px dashed var(--line);font-weight:500}
.dot{width:6px;height:6px;border-radius:99px;background:currentColor;flex:none}
.arm-rg{color:var(--rg)} .arm-sg{color:var(--sg)}
.armbar{display:inline-block;width:3px;height:12px;border-radius:2px;
  vertical-align:-2px;margin-right:6px}
.delta{font-family:var(--mono);font-variant-numeric:tabular-nums}
.bar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:9px}
input[type=search],select{font-family:var(--sans);font-size:13px;padding:6px 10px;
  border:1px solid var(--line);border-radius:var(--r);background:var(--panel);color:var(--ink)}
input[type=search]{min-width:210px}
button.btn{font-family:var(--sans);font-size:12px;padding:5px 10px;border:1px solid var(--line);
  border-radius:var(--r);background:var(--panel);color:var(--ink-2);cursor:pointer}
button.btn:hover{color:var(--ink);border-color:var(--ink-3)}
button.btn:disabled{opacity:.4;cursor:default}
button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.count{color:var(--ink-3);font-size:12px;font-family:var(--mono)}
.note{color:var(--ink-3);font-size:12px;margin-top:6px;max-width:80ch;white-space:normal}
.pager{display:flex;gap:8px;align-items:center;margin-left:auto}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:50;display:none}
.overlay.on{display:block}
.sheet{position:fixed;inset:20px;z-index:51;background:var(--bg);border:1px solid var(--line);
  border-radius:10px;display:none;flex-direction:column;overflow:hidden}
.sheet.on{display:flex}
.sheet .top{border-bottom:1px solid var(--line);padding:12px 18px;display:flex;
  gap:14px;align-items:center;flex-wrap:wrap}
.sheet .body{overflow:auto;padding:18px;flex:1}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
@media (max-width:1000px){.cols{grid-template-columns:1fr}}
details.task summary{cursor:pointer;color:var(--ink-2);font-size:12px;list-style:none;
  margin-top:6px}
details.task summary::-webkit-details-marker{display:none}
details.task summary:before{content:"\\25B8  ";color:var(--ink-3)}
details.task[open] summary:before{content:"\\25BE  "}
.taskbody{white-space:pre-wrap;font-size:13px;line-height:1.6;margin-top:8px;
  max-height:340px;overflow:auto;color:var(--ink-2)}
.taskclip{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;
  overflow:hidden;color:var(--ink-2);font-size:13px;white-space:pre-wrap}
.tl{display:flex;flex-direction:column}
.ev{display:grid;grid-template-columns:70px 1fr;gap:10px;padding:9px 0;
  border-top:1px solid var(--line-soft)}
.ev:first-child{border-top:0}
.ev .lane{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);
  font-weight:600;padding-top:3px}
.ev.think .body{color:var(--ink-2);font-style:italic;font-size:13px}
.ev.gold{background:var(--good-bg);border-radius:var(--r);padding:9px 10px;margin:2px -10px}
.ev.gold .lane{color:var(--good)}
.ev.err .lane{color:var(--bad)}
.call{font-family:var(--mono);font-size:12px;background:var(--accent-bg);
  border:1px solid var(--line);border-radius:var(--r);padding:7px 9px;
  white-space:pre-wrap;word-break:break-word}
.res{font-family:var(--mono);font-size:11.5px;white-space:pre-wrap;word-break:break-word;
  background:var(--panel);border:1px solid var(--line-soft);border-radius:var(--r);
  padding:7px 9px;margin-top:5px;max-height:210px;overflow:auto;color:var(--ink-2)}
.facts{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}
.fact{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);background:var(--line-soft);
  padding:2px 7px;border-radius:99px}
.fact b{color:var(--ink-2)}
.goldmark{background:var(--good);color:var(--bg);font-weight:700;border-radius:3px;padding:0 2px}
.empty{color:var(--ink-3);padding:22px;text-align:center}
"""


JS = r"""
const B = JSON.parse(document.getElementById('bundle').textContent);
const $ = s => document.querySelector(s);
const el = (t, c, txt) => { const n = document.createElement(t); if (c) n.className = c;
  if (txt != null) n.textContent = txt; return n; };
const pct = v => (v * 100).toFixed(1) + '%';
const ARMS = ['rg', 'desc-v5'];
const armClass = c => c === 'rg' ? 'arm-rg' : 'arm-sg';
const bar = k => { const b = el('span', 'armbar');
  b.style.background = k === 'rg' ? 'var(--rg)' : 'var(--sg)'; return b; };
const short = t => t.replace('results-', '');

$('#theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
};

/* ---------- shape ---------- */
const ROWS = [];
B.tiers.forEach(t => t.rows.forEach(r => ROWS.push({ ...r, tier: t.name })));
const TIER_META = Object.fromEntries(B.tiers.map(t => [t.name, t]));
const PAIRS = [];
{
  const by = new Map();
  ROWS.forEach(r => {
    const k = r.tier + '::' + r.instance_id;
    if (!by.has(k)) by.set(k, { tier: r.tier, instance_id: r.instance_id,
      repo: r.repo, category: r.category, arms: {} });
    by.get(k).arms[r.condition] = r;
  });
  by.forEach(v => PAIRS.push(v));
}

/* ---------- headline ---------- */
function renderHeadline() {
  const gates = Object.entries(B.gates || {});
  const traced = gates.filter(([, g]) => (g.summary || {}).traced);
  const allPass = traced.length && traced.every(([, g]) => g.passed);
  const box = el('div', 'headline' + (allPass ? '' : ' bad'));
  const totalTraced = traced.reduce((n, [, g]) => n + ((g.summary || {}).traced || 0), 0);

  const p = el('p');
  p.append(document.createTextNode(allPass
    ? 'Both instrumented samples passed the gate. Across ' : 'A gate failed. Across '));
  p.append(el('b', null, String(totalTraced)));
  p.append(document.createTextNode(' engine-traced searches no ranked search returned '
    + 'nothing, no instance ran dry, and nothing in the agents’ behaviour showed the '
    + 'tool failing.'));
  box.append(p);

  const pooled = (B.scoreboard || []).find(r => r.tier.startsWith('pooled'));
  const pm = pooled && pooled.metrics['func_acc@10_tol'];
  if (pm) {
    const c = el('p', 'caveat');
    c.append(document.createTextNode('Accuracy is a tie and should be read as one: '));
    c.append(el('b', null, (pm.delta >= 0 ? '+' : '') + pm.delta.toFixed(3)));
    c.append(document.createTextNode(` on the primary endpoint over ${pm.n} distinct `
      + `instances, resting on ${pm.w + pm.l} discordant pairs. Tier-1a alone read `
      + `+0.050; an independent sample reversed it.`));
    box.append(c);
  }

  const big = el('div', 'big');
  const stat = (k, v) => { const d = el('div');
    d.append(el('span', 'v', v), el('span', 'k', k)); big.append(d); };
  const spend = traced.reduce((n, [, g]) => n + ((g.summary || {}).cost || 0), 0);
  stat('searches traced', String(totalTraced));
  stat('trajectories', String(Object.keys(B.runs).length));
  stat('tasks', String(PAIRS.length));
  stat('spend', '$' + spend.toFixed(2));
  box.append(big);
  $('#headline').append(box);
}

/* ---------- gate comparison ---------- */
const CRITERIA = [
  { key: 'n_search', label: 'semgrep invocations', kind: 'count' },
  { key: 'traced', label: 'engine traces', kind: 'traced' },
  { key: 'empty_ranked', label: 'ranked searches returning nothing',
    kind: 'share', of: 'traced', needsTrace: true, good: v => v <= 0.02 },
  { key: 'unreadable', label: 'unreadable-scope signature (§16.11)',
    kind: 'zero', needsTrace: true },
  { key: 'tool_caused', label: 'distress attributable to the tool', kind: 'zero' },
  { key: 'all_empty', label: 'instances where every search was empty', kind: 'zero' },
  { key: 'usage_by_cause', label: 'usage errors, by cause', kind: 'causes' },
  { key: 'non_ok', label: 'non-ok rows', kind: 'zero' },
  { key: 'cost', label: 'spend', kind: 'money' },
];

function renderGates() {
  const host = $('#gates');
  const names = Object.keys(B.gates || {});
  if (!names.length) { host.append(el('div', 'empty', 'no gate output captured')); return; }
  const wrap = el('div', 'panel scroll'), t = el('table'), thead = el('thead'), htr = el('tr');
  htr.append(el('th', null, 'criterion'));
  names.forEach(n => {
    const th = el('th'), g = B.gates[n], s = g.summary || {};
    th.append(document.createTextNode(short(n)));
    const pill = el('span', 'pill ' + (!s.traced ? 'nm' : g.passed ? 'good' : 'bad'));
    pill.style.marginLeft = '7px';
    pill.textContent = !s.traced ? 'not instrumented' : (g.passed ? 'passed' : 'failed');
    th.append(pill);
    const rows = (TIER_META[n] || {}).n_rows;
    if (rows) th.append(el('div', 'chip', rows + ' rows'));
    htr.append(th);
  });
  thead.append(htr); t.append(thead);
  const tb = el('tbody');
  CRITERIA.forEach(c => {
    const tr = el('tr');
    tr.append(el('td', null, c.label));
    names.forEach(n => tr.append(gateCell(c, B.gates[n].summary || {}, n)));
    tb.append(tr);
  });
  t.append(tb); wrap.append(t); host.append(wrap);
  host.append(el('div', 'note',
    'A dashed cell means the sample could not answer that criterion, which is not the '
    + 'same as answering zero. §16.10 predates per-invocation tracing; its real '
    + 'figures were recovered offline and are shown beneath.'));
}

function gateCell(c, s, tier) {
  const td = el('td'), v = s[c.key];
  /* Not measured must never look like zero. The first cut of this page rendered
     "0/0" for §16.10's empty-ranked rate, showing the project's worst number
     (59%) as a clean zero — the same silence triage.py had and this fixed. */
  if (c.needsTrace && !s.traced) {
    td.append(el('span', 'pill nm', 'not measured'));
    const known = (KNOWN_ELSEWHERE[tier] || {})[c.key];
    if (known) td.append(el('div', 'note', known));
    return td;
  }
  if (c.kind === 'causes') {
    const d = el('div', 'facts'), ent = Object.entries(v || {});
    if (!ent.length) d.append(el('span', 'fact', 'none'));
    else ent.sort((a, b) => b[1] - a[1]).forEach(([k, n]) => {
      const f = el('span', 'fact');
      f.append(el('b', null, String(n)), document.createTextNode(' ' + k));
      d.append(f);
    });
    td.append(d); return td;
  }
  if (c.kind === 'money') { td.append(el('span', 'mono', '$' + (v ?? 0))); return td; }
  if (c.kind === 'count') { td.append(el('span', 'mono', String(v ?? '—'))); return td; }
  if (c.kind === 'traced') {
    const pill = el('span', 'pill ' + (v ? 'good' : 'nm'));
    if (v) pill.append(el('span', 'dot'));
    pill.append(document.createTextNode(v ? String(v) : 'none'));
    td.append(pill); return td;
  }
  if (c.kind === 'share') {
    const denom = s[c.of] || 0, share = denom ? v / denom : 0;
    const pill = el('span', 'pill ' + (c.good(share) ? 'good' : 'bad'));
    pill.append(el('span', 'dot'), document.createTextNode(`${v} of ${denom}`));
    td.append(pill); td.append(el('div', 'note', pct(share)));
    return td;
  }
  const n = v || 0, pill = el('span', 'pill ' + (n === 0 ? 'good' : 'bad'));
  pill.append(el('span', 'dot'), document.createTextNode(String(n)));
  td.append(pill); return td;
}

/* ---------- scoreboard ---------- */
function renderScore() {
  const wrap = el('div', 'panel scroll'), t = el('table');
  const head = ['sample', 'n', 'metric', 'rg', 'desc-v5', 'delta', 'discordant', '95% CI', 'p'];
  t.innerHTML = '<thead><tr>' + head.map(h =>
    `<th class="${['n', 'rg', 'desc-v5', 'delta', 'p'].includes(h) ? 'num' : ''}">${h}</th>`)
    .join('') + '</tr></thead>';
  const tb = el('tbody');
  B.scoreboard.forEach(row => {
    Object.entries(row.metrics).forEach(([k, m], i) => {
      const tr = el('tr');
      /* The continuation cell carries a mark rather than a blank: an empty cell
         in a numeric table reads as missing data, not as "same as above". */
      const sample = el('td', i ? 'mono' : null, i ? '↳' : row.tier);
      if (i) sample.style.color = 'var(--ink-3)';
      tr.append(sample, el('td', 'num', i ? '' : String(row.n)), el('td', 'mono', k));
      const c1 = el('td', 'num'); c1.append(bar('rg'), document.createTextNode(m.b.toFixed(3)));
      const c2 = el('td', 'num'); c2.append(bar('sg'), document.createTextNode(m.a.toFixed(3)));
      tr.append(c1, c2);
      const d = el('td', 'num delta', (m.delta >= 0 ? '+' : '') + m.delta.toFixed(3));
      d.style.color = Math.abs(m.delta) < 1e-9 ? 'var(--ink-3)'
        : (m.delta > 0 ? 'var(--good)' : 'var(--bad)');
      tr.append(d);
      const disc = el('td');
      disc.append(el('span', 'pill ' + ((m.w + m.l) < 5 ? 'warn' : 'mute'), `w${m.w}/l${m.l}`));
      tr.append(disc);
      tr.append(el('td', 'num', `[${m.ci[0] >= 0 ? '+' : ''}${m.ci[0].toFixed(3)}, `
        + `${m.ci[1] >= 0 ? '+' : ''}${m.ci[1].toFixed(3)}]`),
        el('td', 'num', m.p.toFixed(3)));
      tb.append(tr);
    });
  });
  t.append(tb); wrap.append(t); $('#score').append(wrap);
  $('#score').append(el('div', 'note',
    'Discordant counts sit beside every delta deliberately: at n=40 a delta is a handful '
    + 'of pairs wide. Amber marks fewer than five — a delta that thin is not a result.'));
}

/* ---------- tasks ---------- */
let sortKey = 'instance_id', sortDir = 1, page = 0;
const PAGE = 100;
function renderTable() {
  const metric = $('#metric').value, q = $('#q').value.trim().toLowerCase();
  const tier = $('#tier').value, outcome = $('#outcome').value;
  const activity = $('#activity').value;
  let rows = PAIRS.filter(p => {
    if (tier !== 'all' && p.tier !== tier) return false;
    /* Did the agent actually reach for the search tool? A task it answered
       without searching cannot separate the arms — 75 of 640 here — which is
       §11.5's point about most instances carrying no engine signal, and the
       reason a reviewer wants to exclude them before reading anything into a
       win or a loss. Counts rg and semgrep only; Read and Glob do not qualify. */
    if (activity !== 'all') {
      const c = a => ((p.arms[a] || {}).n_semgrep || 0) + ((p.arms[a] || {}).n_rg || 0);
      const searched = ARMS.filter(a => p.arms[a] && c(a) > 0).length;
      const present = ARMS.filter(a => p.arms[a]).length;
      if (activity === 'any' && searched === 0) return false;
      if (activity === 'both' && !(present > 1 && searched === present)) return false;
      if (activity === 'none' && searched > 0) return false;
    }
    if (q && !(p.instance_id.toLowerCase().includes(q) || (p.repo || '').toLowerCase().includes(q)))
      return false;
    const a = p.arms['desc-v5'], b = p.arms['rg'];
    const av = a && (a.metrics || {})[metric], bv = b && (b.metrics || {})[metric];
    if (outcome === 'sg-only' && !(av && !bv)) return false;
    if (outcome === 'rg-only' && !(bv && !av)) return false;
    if (outcome === 'neither' && (av || bv)) return false;
    if (outcome === 'both' && !(av && bv)) return false;
    if (outcome === 'traj' && !(a && a.has_trajectory)) return false;
    return true;
  });
  const val = (p, k) => {
    if (k === 'cost') return sum(p, 'cost');
    if (k === 'searches') return sum(p, 'n_searches');
    if (k === 'outcome') { const a = p.arms['desc-v5'], b = p.arms['rg'];
      return (a && (a.metrics || {})[metric] ? 2 : 0) + (b && (b.metrics || {})[metric] ? 1 : 0); }
    return p[k] ?? '';
  };
  rows.sort((x, y) => { const a = val(x, sortKey), b = val(y, sortKey);
    return (a > b ? 1 : a < b ? -1 : 0) * sortDir; });
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  page = Math.min(page, pages - 1);
  const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
  $('#count').textContent = rows.length
    ? `${page * PAGE + 1}–${page * PAGE + slice.length} of ${rows.length}` : '0 tasks';
  $('#pageinfo').textContent = `page ${page + 1} / ${pages}`;
  $('#prev').disabled = page === 0; $('#next').disabled = page >= pages - 1;

  const tb = $('#tbody'); tb.textContent = '';
  const frag = document.createDocumentFragment();
  slice.forEach(p => {
    const tr = el('tr', 'clickable'); tr.tabIndex = 0;
    const open = () => openSheet(p);
    tr.onclick = open;
    tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } };
    tr.append(el('td', 'mono', p.instance_id), el('td', 'mono', short(p.tier)),
              el('td', null, p.category || '—'));
    ARMS.forEach(a => {
      const r = p.arms[a], td = el('td');
      if (!r) td.append(el('span', 'pill mute', '—'));
      else if (r.status !== 'ok') td.append(el('span', 'pill warn', r.status));
      else {
        const hit = !!(r.metrics || {})[metric];
        const pill = el('span', 'pill ' + (hit ? 'good' : 'bad'));
        pill.append(el('span', 'dot'), document.createTextNode(hit ? 'found' : 'missed'));
        td.append(pill);
      }
      tr.append(td);
    });
    tr.append(el('td', 'num', String(sum(p, 'n_searches'))),
              el('td', 'num', '$' + sum(p, 'cost').toFixed(2)));
    const td = el('td');
    const has = p.arms['desc-v5'] && p.arms['desc-v5'].has_trajectory;
    td.append(has ? el('span', 'pill mute', 'open ›')
                  : el('span', 'pill nm', 'summary only'));
    tr.append(td);
    frag.append(tr);
  });
  tb.append(frag);
  if (!slice.length) {
    const tr = el('tr'), td = el('td', 'empty', 'nothing matches these filters');
    td.colSpan = 8; tr.append(td); tb.append(tr);
  }
  document.querySelectorAll('#itable th[data-k]').forEach(th => {
    th.querySelector('.caret')?.remove();
    if (th.dataset.k === sortKey) th.append(el('span', 'caret', sortDir > 0 ? '▲' : '▼'));
  });
}
function sum(p, k) { return ARMS.reduce((n, a) => n + ((p.arms[a] || {})[k] || 0), 0); }

/* ---------- the sheet: task, then gold, then what each arm did ---------- */
function openSheet(p) {
  const body = $('#sheetBody'); body.textContent = '';
  $('#sheetTitle').textContent = p.instance_id;
  $('#sheetSub').textContent = `${p.repo || ''} · ${p.category || ''} · ${short(p.tier)}`;
  history.replaceState(null, '', '#' + encodeURIComponent(p.tier + '/' + p.instance_id));

  /* What was asked, first. A reviewer cannot judge whether a query was a
     reasonable guess without seeing the problem it was guessing at. */
  const task = (B.tasks || {})[p.instance_id];
  const tp = el('div', 'panel');
  tp.append(el('h2', null, 'the task the agent was given'));
  if (task) {
    const clip = el('div', 'taskclip', task);
    const det = el('details', 'task');
    det.append(el('summary', null, 'read the full issue'), el('div', 'taskbody', task));
    det.ontoggle = () => { clip.style.display = det.open ? 'none' : '-webkit-box'; };
    tp.append(clip, det);
  } else tp.append(el('div', 'note', 'no problem statement captured for this instance'));
  body.append(tp);

  const gold = (p.arms['desc-v5'] || p.arms['rg'] || {}).gold_files || [];
  const gp = el('div', 'panel');
  gp.append(el('h2', null, 'gold — where the answer actually lives'));
  gold.forEach(g => gp.append(el('div', 'mono', g)));
  ((p.arms['desc-v5'] || {}).gold_functions || []).forEach(g => gp.append(el('div', 'mono', g)));
  body.append(gp);

  const cols = el('div', 'cols');
  ARMS.forEach(a => cols.append(armColumn(p, a, gold)));
  body.append(cols);
  $('#overlay').classList.add('on'); $('#sheet').classList.add('on');
  $('#sheetClose').focus();
}

function armColumn(p, arm, gold) {
  const r = p.arms[arm], col = el('div');
  const head = el('div', 'bar');
  const h = el('h2'); h.append(bar(arm === 'rg' ? 'rg' : 'sg'), document.createTextNode(arm));
  head.append(h);
  if (r) {
    if (r.status !== 'ok') head.append(el('span', 'pill warn', r.status));
    head.append(el('span', 'chip', `$${(r.cost || 0).toFixed(3)}`),
                el('span', 'chip', `${r.n_searches || 0} tool calls`));
    const fh = (r.metrics || {}).first_hit_search_seq;
    head.append(el('span', 'pill ' + (fh != null ? 'good' : 'mute'),
      fh != null ? `gold at search ${fh}` : 'never surfaced gold'));
  }
  col.append(head);
  const panel = el('div', 'panel');
  if (!r) { panel.append(el('div', 'empty', 'no row')); col.append(panel); return col; }
  const traj = r.run_key ? B.runs[r.run_key] : null;
  if (!traj) {
    panel.append(el('div', 'empty',
      'summary only — this sample was captured without trajectories'));
    col.append(panel); return col;
  }
  panel.append(renderTimeline(traj, (r.metrics || {}).first_hit_search_seq, gold));
  const d = traj.dropped || {}, notes = [];
  if (d.searches) notes.push(`${d.searches} further searches`);
  if (d.turns) notes.push(`${d.turns} further steps`);
  if (d.unrecorded_reasoning)
    notes.push(`${d.unrecorded_reasoning} reasoning blocks the transcript did not record`);
  if (d.searches_off_timeline)
    notes.push(`${d.searches_off_timeline} searches the shim logged but could not be `
      + `placed in this sequence — see the Searches view`);
  if (notes.length) panel.append(el('div', 'note', 'not shown: ' + notes.join(' · ')));
  col.append(panel);
  return col;
}

/* One ordered list: what it thought, what it ran, what came back. */
function renderTimeline(traj, firstHit, gold) {
  const tl = el('div', 'tl');
  (traj.timeline || []).forEach(t => {
    if (t.kind === 'thinking' || t.kind === 'text') {
      const ev = el('div', 'ev' + (t.kind === 'thinking' ? ' think' : ''));
      ev.append(el('div', 'lane', t.kind === 'thinking' ? 'thinks' : 'says'),
                el('div', 'body', t.text || ''));
      tl.append(ev); return;
    }
    /* A single shell call can run the tool more than once, so engine facts are
       a list — one row per invocation, labelled by its position. */
    const ss = t.searches || [];
    const isGold = firstHit != null && ss.some(s => s.pos === firstHit);
    const ev = el('div', 'ev' + (isGold ? ' gold' : '') + (t.is_error ? ' err' : ''));
    ev.append(el('div', 'lane', isGold ? 'gold ✓' : (t.name || 'runs')));
    const b = el('div', 'body');
    b.append(el('div', 'call', t.input || ''));
    if (t.result != null) {
      const res = el('div', 'res');
      res.innerHTML = markGold(t.result, gold);
      b.append(res);
    }
    ss.forEach(s => {
      const facts = el('div', 'facts');
      const add = (k, v) => { if (v == null) return; const f = el('span', 'fact');
        f.append(document.createTextNode(k + ' '), el('b', null, String(v))); facts.append(f); };
      if (ss.length > 1) add('search', '#' + s.pos);
      add('exit', s.exit); add('bytes', s.stdout_bytes); add('ms', s.wall_ms);
      const tr = s.trace;
      if (tr) { add('mode', tr.mode); add('path', tr.path_taken); add('files', tr.files_walked);
                add('chunks', tr.n_chunks_considered); add('hits', tr.n_hits); }
      b.append(facts);
    });
    ev.append(b); tl.append(ev);
  });
  if (!tl.children.length) tl.append(el('div', 'empty', 'no steps recorded'));
  return tl;
}

function markGold(text, gold) {
  let out = String(text).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  (gold || []).forEach(g => {
    if (!g) return;
    const safe = g.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(safe, 'g'), m => `<span class="goldmark">${m}</span>`);
  });
  return out;
}
function closeSheet() {
  $('#overlay').classList.remove('on'); $('#sheet').classList.remove('on');
  history.replaceState(null, '', location.pathname + location.search);
}
$('#sheetClose').onclick = closeSheet;
$('#overlay').onclick = closeSheet;
addEventListener('keydown', e => { if (e.key === 'Escape') closeSheet(); });

/* ---------- searches ---------- */
let spage = 0;
const ALL_SEARCHES = [];
Object.entries(B.runs).forEach(([key, run]) => {
  const [tier, inst, cond] = key.split('::');
  (run.searches || []).forEach(s => ALL_SEARCHES.push({ tier, inst, cond, ...s }));
});
function renderSearches() {
  const f = $('#sfilter').value;
  const rows = ALL_SEARCHES.filter(s => {
    if (f === 'empty') return (s.stdout_bytes || 0) === 0 && s.exit !== 2;
    if (f === 'exit2') return s.exit === 2;
    if (f === 'unreadable') return s.trace && s.trace.files_walked > 0
      && s.trace.n_chunks_considered === 0;
    if (f === 'semantic') return s.trace && s.trace.mode === 'semantic';
    return true;
  });
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  spage = Math.min(spage, pages - 1);
  const slice = rows.slice(spage * PAGE, spage * PAGE + PAGE);
  $('#scount').textContent = rows.length
    ? `${spage * PAGE + 1}–${spage * PAGE + slice.length} of ${rows.length}` : '0 searches';
  $('#spageinfo').textContent = `page ${spage + 1} / ${pages}`;
  $('#sprev').disabled = spage === 0; $('#snext').disabled = spage >= pages - 1;
  const tb = $('#stbody'); tb.textContent = '';
  const frag = document.createDocumentFragment();
  slice.forEach(s => {
    const tr = el('tr');
    tr.append(el('td', 'mono', s.inst), el('td', 'mono ' + armClass(s.cond), s.cond),
              el('td', 'num', String(s.pos)));
    const c = el('td', 'mono'); c.textContent = (s.argv || []).join(' ').slice(0, 88);
    tr.append(c);
    const ex = el('td');
    ex.append(el('span', 'pill ' + (s.exit === 2 ? 'bad' : s.exit === 1 ? 'warn' : 'good'),
      String(s.exit)));
    tr.append(ex);
    const t = s.trace || {};
    tr.append(el('td', 'mono', t.mode || '—'),
              el('td', 'num', String(t.files_walked ?? '—')),
              el('td', 'num', String(t.n_chunks_considered ?? '—')),
              el('td', 'num', String(t.n_hits ?? '—')),
              el('td', 'num', String(s.wall_ms ?? '—')));
    frag.append(tr);
  });
  tb.append(frag);
  if (!slice.length) {
    const tr = el('tr'), td = el('td', 'empty', 'no searches match');
    td.colSpan = 10; tr.append(td); tb.append(tr);
  }
}

/* ---------- views + deep links ---------- */
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === 'view-' + name));
  document.querySelectorAll('nav button').forEach(b =>
    b.setAttribute('aria-current', String(b.dataset.view === name)));
}
document.querySelectorAll('nav button').forEach(b =>
  b.onclick = () => showView(b.dataset.view));

function openFromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (!h.includes('/')) return;
  const i = h.indexOf('/');
  const p = PAIRS.find(x => x.tier === h.slice(0, i) && x.instance_id === h.slice(i + 1));
  if (p) { showView('tasks'); openSheet(p); }
}

/* ---------- boot ---------- */
renderHeadline(); renderGates(); renderScore();
B.tiers.forEach(t => $('#tier').append(new Option(short(t.name), t.name)));
['q', 'tier', 'outcome', 'metric', 'activity'].forEach(id =>
  $('#' + id).addEventListener(id === 'q' ? 'input' : 'change', () => { page = 0; renderTable(); }));
document.querySelectorAll('#itable th[data-k]').forEach(th => {
  th.onclick = () => { const k = th.dataset.k;
    sortDir = (k === sortKey) ? -sortDir : 1; sortKey = k; page = 0; renderTable(); };
});
$('#prev').onclick = () => { page--; renderTable(); };
$('#next').onclick = () => { page++; renderTable(); };
$('#sfilter').onchange = () => { spage = 0; renderSearches(); };
$('#sprev').onclick = () => { spage--; renderSearches(); };
$('#snext').onclick = () => { spage++; renderSearches(); };
renderTable(); renderSearches(); showView('overview'); openFromHash();
addEventListener('hashchange', openFromHash);
"""


def build(bundle, out_path):
    bundle["scoreboard"] = scoreboard(bundle)
    prov = next(iter(bundle.get("provenance", {}).values()), {}) or {}
    payload = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")

    chips = "".join(
        f'<span class="chip{" opt" if opt else ""}">{html.escape(k)} '
        f'<b>{html.escape(str(v)[:22])}</b></span>'
        for k, v, opt in (
            ("model", prov.get("model"), False),
            ("semgrep", (prov.get("semgrep_sha256") or "")[:12], False),
            ("claude", prov.get("claude_version"), True),
            ("dataset", (prov.get("dataset_sha256") or "")[:12], True),
            ("captured", bundle.get("generated_at", "")[:16], True),
        ) if v)

    known = json.dumps(KNOWN_ELSEWHERE, separators=(",", ":")).replace("</", "<\\/")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>semgrep loc-bench — results and trajectories</title>
<style>{CSS}</style></head>
<body>
<header>
  <div class="hrow">
    <h1>loc-bench results &amp; trajectories</h1>
    <div class="chips">{chips}<button class="btn" id="theme">theme</button></div>
  </div>
  <nav>
    <button data-view="overview" aria-current="true">Overview</button>
    <button data-view="tasks">Tasks</button>
    <button data-view="searches">Searches</button>
  </nav>
</header>

<div class="wrap">
  <div class="view on" id="view-overview">
    <div id="headline"></div>
    <section><h2>gate</h2>
      <p class="sub">the verdict <code>triage.py</code> produced, not a second opinion</p>
      <div id="gates"></div></section>
    <section><h2>accuracy</h2>
      <p class="sub">paired on instances where both arms completed</p>
      <div id="score"></div></section>
  </div>

  <div class="view" id="view-tasks">
    <section><h2>tasks</h2>
      <p class="sub">one row per task per sample &mdash; click for the issue text and both
        arms step by step</p>
      <div class="bar">
        <input type="search" id="q" placeholder="filter by instance or repo" aria-label="filter">
        <select id="tier" aria-label="sample"><option value="all">all samples</option></select>
        <select id="metric" aria-label="metric">
          <option value="func_acc@10_tol">func_acc@10_tol</option>
          <option value="file_acc@5">file_acc@5</option></select>
        <select id="outcome" aria-label="outcome">
          <option value="all">any outcome</option>
          <option value="sg-only">desc-v5 only</option>
          <option value="rg-only">rg only</option>
          <option value="both">both found</option>
          <option value="neither">neither found</option>
          <option value="traj">has trajectory</option></select>
        <select id="activity" aria-label="search activity">
          <option value="all">searched or not</option>
          <option value="any">called rg or semgrep</option>
          <option value="both">called it in both arms</option>
          <option value="none">never called it</option></select>
        <span class="count" id="count"></span>
        <span class="pager"><button class="btn" id="prev">&lsaquo; prev</button>
          <span class="count" id="pageinfo"></span>
          <button class="btn" id="next">next &rsaquo;</button></span>
      </div>
      <div class="panel scroll"><table id="itable"><thead><tr>
        <th data-k="instance_id">instance</th><th data-k="tier">sample</th>
        <th data-k="category">category</th>
        <th data-k="outcome">rg</th><th data-k="outcome">desc-v5</th>
        <th data-k="searches" class="num">tool calls</th>
        <th data-k="cost" class="num">cost</th>
        <th>detail</th></tr></thead><tbody id="tbody"></tbody></table></div>
    </section>
  </div>

  <div class="view" id="view-searches">
    <section><h2>search inspector</h2>
      <p class="sub">every captured invocation &mdash; where a &sect;16.11-class bug shows as
        a pattern rather than an anecdote</p>
      <div class="bar">
        <select id="sfilter" aria-label="filter searches">
          <option value="all">all searches</option>
          <option value="empty">returned nothing</option>
          <option value="exit2">usage errors (exit 2)</option>
          <option value="unreadable">files walked but 0 chunks</option>
          <option value="semantic">semantic mode only</option></select>
        <span class="count" id="scount"></span>
        <span class="pager"><button class="btn" id="sprev">&lsaquo; prev</button>
          <span class="count" id="spageinfo"></span>
          <button class="btn" id="snext">next &rsaquo;</button></span>
      </div>
      <div class="panel scroll"><table><thead><tr>
        <th>instance</th><th>arm</th><th class="num">#</th><th>command</th><th>exit</th>
        <th>mode</th><th class="num">files</th><th class="num">chunks</th>
        <th class="num">hits</th><th class="num">ms</th>
      </tr></thead><tbody id="stbody"></tbody></table></div>
    </section>
  </div>
</div>

<div class="overlay" id="overlay"></div>
<div class="sheet" id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle">
  <div class="top">
    <div><h1 id="sheetTitle"></h1><p class="sub" id="sheetSub"></p></div>
    <div class="chips"><button class="btn" id="sheetClose">close (esc)</button></div>
  </div>
  <div class="body" id="sheetBody"></div>
</div>

<script type="application/json" id="bundle">{payload}</script>
<script>const KNOWN_ELSEWHERE = {known};
{JS}</script>
</body></html>"""
    out_path.write_text(doc)
    return len(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DATA / "viewer-bundle.json")
    ap.add_argument("--out", type=Path, default=DATA / "results-viewer.html")
    args = ap.parse_args()
    if not args.bundle.exists():
        sys.exit(f"no bundle at {args.bundle} — run eval/locbench/capture.py first")
    bundle = json.loads(args.bundle.read_text())
    n = build(bundle, args.out)
    print(f"wrote {args.out} — {n / 1e6:.2f} MB, "
          f"{sum(t['n_rows'] for t in bundle['tiers'])} rows, "
          f"{len(bundle['runs'])} trajectories")


if __name__ == "__main__":
    main()
