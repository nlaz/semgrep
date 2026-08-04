#!/usr/bin/env python3
"""Render a capture bundle as one self-contained HTML page.

    python3 eval/locbench/capture.py
    python3 eval/locbench/viewer.py            # -> eval/data/locbench/results-viewer.html

No external fonts, scripts, styles, or fetches: the page opens from `file://`
with the network off. The bundle is inlined; statistics are computed here, at
build time, by the same functions `ab_analyze.py` uses for the published
tables, so the page cannot quietly disagree with them.

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


def scoreboard(bundle):
    """Paired accuracy per tier and pooled, with discordant counts.

    Discordant counts travel with every delta on purpose. At n=40 a delta is
    four pairs wide, and a table showing +0.050 without w2/l0 beside it invites
    exactly the misreading §18.6 caught when an independent sample reversed the
    sign.
    """
    out = []
    pooled = {}
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
        cost = {a: mean([r["cost"] for r in tier["rows"]
                         if r["condition"] == a and r["status"] == "ok"]) for a in ARMS}
        srch = {a: mean([r["n_searches"] or 0 for r in tier["rows"]
                         if r["condition"] == a and r["status"] == "ok"]) for a in ARMS}
        row["cost"], row["searches"] = cost, srch
        out.append(row)

    if pooled:
        row = {"tier": "pooled (distinct instances)", "n": len(pooled), "metrics": {}}
        for key in (PRIMARY, SECONDARY):
            pairs = [v[key] for v in pooled.values() if key in v]
            row["metrics"][key] = stat_block(pairs)
        out.append(row)
    return out


def stat_block(pairs):
    if not pairs:
        return {"a": 0, "b": 0, "delta": 0, "w": 0, "l": 0, "p": 1.0, "ci": [0, 0]}
    n = len(pairs)
    a = sum(1 for x, _ in pairs if x) / n      # desc-v5
    b = sum(1 for _, y in pairs if y) / n      # rg
    blk = {"a": a, "b": b, "delta": a - b, "n": n}
    if mcnemar:
        w, l, p = mcnemar([(int(x), int(y)) for x, y in pairs])
        blk.update(w=w, l=l, p=p)
        pt, lo, hi = boot_ci([(int(x), int(y)) for x, y in pairs])
        blk["ci"] = [lo, hi]
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
  --accent:#1F5FA8;
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
  --accent:#4A86D0;
}}
:root[data-theme="dark"]{
  --bg:#14171C; --panel:#1A1E25; --line:#2C323C; --line-soft:#232830;
  --ink:#E8EAED; --ink-2:#A8B0BC; --ink-3:#78818F;
  --rg:#4A86D0; --sg:#C4703A;
  --good:#1F9E92; --warn:#B58E10; --bad:#D6455F;
  --good-bg:#0F2B2A; --warn-bg:#2C2612; --bad-bg:#331520;
  --accent:#4A86D0;
}
:root[data-theme="light"]{
  --bg:#FBFAF7; --panel:#FFFFFF; --line:#E3E0D8; --line-soft:#EFECE5;
  --ink:#1B1D22; --ink-2:#4B5058; --ink-3:#797F88;
  --rg:#1F5FA8; --sg:#C25A18;
  --good:#0C968C; --warn:#A67C00; --bad:#C62B4E;
  --good-bg:#E6F4F2; --warn-bg:#F7F0DE; --bad-bg:#FAE7EC;
  --accent:#1F5FA8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1400px;margin:0 auto;padding:0 20px 80px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
h2{font-size:15px;margin:0 0 4px;letter-spacing:.02em;text-transform:uppercase;color:var(--ink-2)}
.sub{color:var(--ink-3);font-size:13px;margin:2px 0 0}
header{position:sticky;top:0;z-index:20;background:var(--bg);
  border-bottom:1px solid var(--line);padding:14px 0 12px;margin-bottom:22px}
.hrow{max-width:1400px;margin:0 auto;padding:0 20px;display:flex;gap:16px;
  align-items:baseline;flex-wrap:wrap}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.chip{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:99px;
  border:1px solid var(--line);color:var(--ink-2);background:var(--panel);white-space:nowrap}
.chip b{color:var(--ink);font-weight:600}
section{margin-bottom:34px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
  padding:16px;margin-top:10px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line-soft);
  white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink-3);
  font-weight:600;position:sticky;top:0;background:var(--panel);cursor:pointer;
  user-select:none}
th.no{cursor:default}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono)}
tbody tr:hover{background:var(--line-soft)}
tbody tr.clickable{cursor:pointer}
.mono{font-family:var(--mono)}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
  padding:2px 8px;border-radius:99px;font-family:var(--mono)}
.pill.good{background:var(--good-bg);color:var(--good)}
.pill.warn{background:var(--warn-bg);color:var(--warn)}
.pill.bad{background:var(--bad-bg);color:var(--bad)}
.pill.mute{background:var(--line-soft);color:var(--ink-3)}
.dot{width:6px;height:6px;border-radius:99px;background:currentColor;flex:none}
.arm-rg{color:var(--rg)} .arm-sg{color:var(--sg)}
.armbar{display:inline-block;width:3px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:6px}
.delta{font-family:var(--mono);font-variant-numeric:tabular-nums}
.gates{display:grid;gap:2px}
.gate{display:grid;grid-template-columns:1fr auto auto;gap:12px;align-items:center;
  padding:7px 10px;border-bottom:1px solid var(--line-soft)}
.gate:last-child{border-bottom:0}
.gate .lab{color:var(--ink-2)}
.gate .val{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
input[type=search],select{font-family:var(--sans);font-size:13px;padding:6px 10px;
  border:1px solid var(--line);border-radius:var(--r);background:var(--panel);color:var(--ink)}
input[type=search]{min-width:230px}
button{font-family:var(--sans);font-size:12px;padding:5px 10px;border:1px solid var(--line);
  border-radius:var(--r);background:var(--panel);color:var(--ink-2);cursor:pointer}
button:hover{color:var(--ink);border-color:var(--ink-3)}
button:focus-visible,tr:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.count{color:var(--ink-3);font-size:12px;font-family:var(--mono)}
/* drill-down */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:50;display:none}
.overlay.on{display:block}
.sheet{position:fixed;inset:24px;z-index:51;background:var(--bg);border:1px solid var(--line);
  border-radius:10px;display:none;flex-direction:column;overflow:hidden}
.sheet.on{display:flex}
.sheet header{position:static;border-bottom:1px solid var(--line);margin:0;padding:14px 18px}
.sheet .body{overflow:auto;padding:18px;flex:1}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.turn{border-left:2px solid var(--line);padding:6px 0 6px 12px;margin:8px 0}
.turn.think{border-left-color:var(--ink-3);color:var(--ink-2);font-style:italic}
.turn.search{border-left-color:var(--accent)}
.turn.gold{border-left-color:var(--good);background:var(--good-bg);border-radius:0 var(--r) var(--r) 0}
.turn .k{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);
  font-weight:600;margin-bottom:3px}
pre{font-family:var(--mono);font-size:12px;margin:6px 0 0;white-space:pre-wrap;
  word-break:break-word;background:var(--panel);border:1px solid var(--line-soft);
  border-radius:var(--r);padding:8px 10px;max-height:260px;overflow:auto}
.facts{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.fact{font-family:var(--mono);font-size:10.5px;color:var(--ink-2);background:var(--line-soft);
  padding:2px 7px;border-radius:99px}
.fact b{color:var(--ink)}
.goldmark{background:var(--good-bg);color:var(--good);font-weight:700;border-radius:3px;padding:0 2px}
.empty{color:var(--ink-3);padding:20px;text-align:center}
.note{color:var(--ink-3);font-size:12px;margin-top:8px}
"""


JS = r"""
const B = JSON.parse(document.getElementById('bundle').textContent);
const $ = s => document.querySelector(s);
const el = (t, c, txt) => { const n = document.createElement(t); if (c) n.className = c;
  if (txt != null) n.textContent = txt; return n; };
const esc = s => String(s == null ? '' : s);
const pct = v => (v * 100).toFixed(1) + '%';
const armClass = c => c === 'rg' ? 'arm-rg' : 'arm-sg';

/* ---------- theme ---------- */
$('#theme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
};

/* ---------- flatten rows ---------- */
const ROWS = [];
B.tiers.forEach(t => t.rows.forEach(r => ROWS.push({ ...r, tier: t.name })));
const TIERS = B.tiers.map(t => t.name);

/* Pair the two arms per (tier, instance) so the table is one row per task. */
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

/* ---------- gates ---------- */
function renderGates() {
  const host = $('#gates');
  const names = Object.keys(B.gates || {});
  if (!names.length) { host.append(el('div', 'empty', 'no gate output captured')); return; }
  names.forEach(n => {
    const g = B.gates[n], p = el('div', 'panel');
    const head = el('div', 'bar');
    head.append(el('h2', null, n));
    const verdict = el('span', 'pill ' + (g.passed ? 'good' : 'bad'));
    verdict.append(el('span', 'dot'), document.createTextNode(g.passed ? 'GATE PASSED' : 'GATE FAILED'));
    head.append(verdict);
    p.append(head);
    const s = g.summary || {}, box = el('div', 'gates');
    const line = (lab, val, state, note) => {
      const row = el('div', 'gate');
      row.append(el('div', 'lab', lab));
      // `val` may be a node (the usage-error breakdown renders as chips, since
      // a dumped JSON object is unreadable at exactly the moment a reviewer
      // most wants to read it).
      row.append(val instanceof Node ? val : el('div', 'val', val));
      const pill = el('span', 'pill ' + state);
      pill.append(el('span', 'dot'), document.createTextNode(note));
      row.append(pill); box.append(row);
    };
    const causeChips = obj => {
      const d = el('div', 'facts');
      const ent = Object.entries(obj || {});
      if (!ent.length) { d.append(el('span', 'fact', 'none')); return d; }
      ent.sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
        const f = el('span', 'fact');
        f.append(el('b', null, String(v)), document.createTextNode(' ' + k));
        d.append(f);
      });
      return d;
    };
    const emptyShare = s.traced ? (s.empty_ranked / s.traced) : null;
    line('semgrep invocations', String(s.n_search ?? '—'), 'mute', 'context');
    line('engine traces', String(s.traced ?? '—'), s.traced ? 'good' : 'bad',
         s.traced ? 'instrumented' : 'no trace');
    if (emptyShare !== null)
      line('ranked searches returning nothing', s.empty_ranked + '/' + s.traced,
           emptyShare <= 0.02 ? 'good' : 'bad', pct(emptyShare));
    line('unreadable-scope signature (§16.11)', String(s.unreadable ?? '—'),
         (s.unreadable || 0) === 0 ? 'good' : 'bad', (s.unreadable || 0) === 0 ? 'absent' : 'present');
    line('distress attributable to the tool', String(s.tool_caused ?? '—'),
         (s.tool_caused || 0) === 0 ? 'good' : 'bad', (s.tool_caused || 0) === 0 ? 'none' : 'present');
    line('instances where every search was empty', String(s.all_empty ?? '—'),
         (s.all_empty || 0) === 0 ? 'good' : 'bad', (s.all_empty || 0) === 0 ? 'none' : 'present');
    line('usage errors', causeChips(s.usage_by_cause), 'mute', 'by cause');
    line('non-ok rows', String(s.non_ok ?? '—'), (s.non_ok || 0) === 0 ? 'good' : 'warn',
         (s.non_ok || 0) === 0 ? 'clean' : 'see rows');
    line('spend', '$' + (s.cost ?? 0), 'mute', 'usd');
    p.append(box);
    if ((g.failures || []).length) {
      const n = el('div', 'note', 'failed: ' + g.failures.map(f => f[0]).join(' · '));
      p.append(n);
    }
    host.append(p);
  });
}

/* ---------- scoreboard ---------- */
function renderScore() {
  const host = $('#score');
  const wrap = el('div', 'panel scroll'), t = el('table');
  t.innerHTML = '<thead><tr>' +
    ['sample', 'n', 'metric', 'rg', 'desc-v5', 'delta', 'discordant', '95% CI', 'p'].
      map(h => `<th class="no${['n','rg','desc-v5','delta','p'].includes(h) ? ' num' : ''}">${h}</th>`).join('') +
    '</tr></thead>';
  const tb = el('tbody');
  B.scoreboard.forEach(row => {
    Object.entries(row.metrics).forEach(([k, m], i) => {
      const tr = el('tr');
      tr.append(el('td', null, i === 0 ? row.tier : ''),
                el('td', 'num', i === 0 ? String(row.n) : ''),
                el('td', 'mono', k));
      const rg = el('td', 'num'); rg.append(bar('rg'), document.createTextNode(m.b.toFixed(3)));
      const sg = el('td', 'num'); sg.append(bar('sg'), document.createTextNode(m.a.toFixed(3)));
      tr.append(rg, sg);
      const d = el('td', 'num delta', (m.delta >= 0 ? '+' : '') + m.delta.toFixed(3));
      d.style.color = Math.abs(m.delta) < 1e-9 ? 'var(--ink-3)'
        : (m.delta > 0 ? 'var(--good)' : 'var(--bad)');
      tr.append(d);
      const disc = el('td', 'mono');
      const pill = el('span', 'pill ' + ((m.w + m.l) < 5 ? 'warn' : 'mute'));
      pill.textContent = `w${m.w}/l${m.l}`;
      disc.append(pill);
      tr.append(disc,
        el('td', 'num', `[${m.ci[0] >= 0 ? '+' : ''}${m.ci[0].toFixed(3)}, ${m.ci[1] >= 0 ? '+' : ''}${m.ci[1].toFixed(3)}]`),
        el('td', 'num', m.p.toFixed(3)));
      tb.append(tr);
    });
  });
  t.append(tb); wrap.append(t); host.append(wrap);
  host.append(el('div', 'note',
    'Discordant counts sit beside every delta deliberately: at n=40 a delta is a handful of pairs wide. '
    + 'Amber marks fewer than five discordant pairs — a delta that thin is not a result.'));
}
function bar(kind) { const b = el('span', 'armbar');
  b.style.background = kind === 'rg' ? 'var(--rg)' : 'var(--sg)'; return b; }

/* ---------- instance table ---------- */
let sortKey = 'instance_id', sortDir = 1;
function renderTable() {
  const q = $('#q').value.trim().toLowerCase();
  const tier = $('#tier').value, outcome = $('#outcome').value;
  const metric = $('#metric').value;
  let rows = PAIRS.filter(p => {
    if (tier !== 'all' && p.tier !== tier) return false;
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
  $('#count').textContent = `${rows.length} of ${PAIRS.length} tasks`;
  const tb = $('#tbody'); tb.textContent = '';
  const frag = document.createDocumentFragment();
  rows.forEach(p => {
    const tr = el('tr', 'clickable'); tr.tabIndex = 0;
    const open = () => openSheet(p);
    tr.onclick = open;
    tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } };
    // The sample column is not decoration: an instance can appear in several
    // samples (tier-1a and tier-1b overlap by 7, and both overlap §16.10), so
    // without it the table looks like it is repeating rows.
    tr.append(el('td', 'mono', p.instance_id),
              el('td', 'mono', p.tier.replace('results-', '')),
              el('td', null, p.category || '—'));
    ARMS_JS.forEach(a => {
      const r = p.arms[a], td = el('td');
      if (!r) { td.append(el('span', 'pill mute', '—')); }
      else if (r.status !== 'ok') { td.append(el('span', 'pill warn', r.status)); }
      else {
        const hit = !!(r.metrics || {})[metric];
        const pill = el('span', 'pill ' + (hit ? 'good' : 'bad'));
        pill.append(el('span', 'dot'), document.createTextNode(hit ? 'found' : 'missed'));
        td.append(pill);
      }
      tr.append(td);
    });
    tr.append(el('td', 'num', fmt(sum(p, 'n_searches'))),
              el('td', 'num', '$' + sum(p, 'cost').toFixed(2)));
    const t = el('td');
    if (p.arms['desc-v5'] && p.arms['desc-v5'].has_trajectory)
      t.append(el('span', 'pill mute', 'open'));
    tr.append(t);
    frag.append(tr);
  });
  tb.append(frag);
  if (!rows.length) {
    const tr = el('tr'), td = el('td', 'empty', 'nothing matches');
    td.colSpan = 8; tr.append(td); tb.append(tr);
  }
}
const ARMS_JS = ['rg', 'desc-v5'];
function sum(p, k) { return ARMS_JS.reduce((n, a) => n + ((p.arms[a] || {})[k] || 0), 0); }
function fmt(n) { return Number.isFinite(n) ? String(n) : '—'; }

/* ---------- drill-down ---------- */
function openSheet(p) {
  const body = $('#sheetBody'); body.textContent = '';
  $('#sheetTitle').textContent = p.instance_id;
  $('#sheetSub').textContent = `${p.repo || ''} · ${p.category || ''} · ${p.tier}`;
  const gold = (p.arms['desc-v5'] || p.arms['rg'] || {}).gold_files || [];
  const gbox = el('div', 'panel');
  gbox.append(el('h2', null, 'gold'));
  gold.forEach(g => { const d = el('div', 'mono'); d.textContent = g; gbox.append(d); });
  ((p.arms['desc-v5'] || {}).gold_functions || []).forEach(g =>
    gbox.append(el('div', 'mono', g)));
  body.append(gbox);

  const cols = el('div', 'cols');
  ARMS_JS.forEach(a => cols.append(armColumn(p, a, gold)));
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
    const ok = r.status === 'ok';
    head.append(el('span', 'pill ' + (ok ? 'mute' : 'warn'), r.status));
    head.append(el('span', 'chip', `$${(r.cost || 0).toFixed(3)}`));
    head.append(el('span', 'chip', `${r.n_searches || 0} searches`));
  }
  col.append(head);
  if (!r) { col.append(el('div', 'empty', 'no row')); return col; }

  const traj = r.run_key ? B.runs[r.run_key] : null;
  const firstHit = (r.metrics || {}).first_hit_search_seq;
  if (!traj) {
    // Node.append() returns undefined, so it cannot be chained — building the
    // panel first is not style, it is the difference between this rendering
    // and throwing.
    const p = el('div', 'panel');
    p.append(el('div', 'empty', 'summary only — this tier was captured without trajectories'));
    col.append(p);
    return col;
  }
  const panel = el('div', 'panel');
  traj.searches.forEach(s => {
    const isGold = firstHit != null && s.pos === firstHit;
    const d = el('div', 'turn search' + (isGold ? ' gold' : ''));
    d.append(el('div', 'k', `#${s.pos} ${s.tool}${isGold ? ' · first search to surface gold' : ''}`));
    const cmd = el('pre'); cmd.textContent = (s.argv || []).join(' ');
    d.append(cmd);
    const facts = el('div', 'facts');
    const add = (k, v) => { const f = el('span', 'fact'); f.append(document.createTextNode(k + ' '));
      f.append(el('b', null, String(v))); facts.append(f); };
    add('exit', s.exit); add('bytes', s.stdout_bytes); add('ms', s.wall_ms);
    if (s.trace) { const t = s.trace;
      add('mode', t.mode); add('path', t.path_taken);
      add('files', t.files_walked); add('chunks', t.n_chunks_considered); add('hits', t.n_hits); }
    d.append(facts);
    if (s.out && s.out.trim()) {
      const o = el('pre'); o.innerHTML = markGold(s.out, gold); d.append(o);
    }
    panel.append(d);
  });
  if (traj.dropped && (traj.dropped.searches || traj.dropped.turns))
    panel.append(el('div', 'note',
      `truncated: ${traj.dropped.searches} searches, ${traj.dropped.turns} turns not shown`));
  col.append(panel);

  if (traj.turns.length) {
    const tp = el('div', 'panel');
    tp.append(el('h2', null, 'agent turns'));
    traj.turns.forEach(t => {
      const d = el('div', 'turn ' + (t.kind === 'thinking' ? 'think' : ''));
      d.append(el('div', 'k', t.kind + (t.name ? ' · ' + t.name : '')));
      d.append(document.createTextNode(t.text || t.input || ''));
      tp.append(d);
    });
    col.append(tp);
  }
  return col;
}
function markGold(text, gold) {
  let out = text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  (gold || []).forEach(g => {
    if (!g) return;
    const safe = g.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(safe, 'g'), m => `<span class="goldmark">${m}</span>`);
  });
  return out;
}
function closeSheet() { $('#overlay').classList.remove('on'); $('#sheet').classList.remove('on'); }
$('#sheetClose').onclick = closeSheet;
$('#overlay').onclick = closeSheet;
addEventListener('keydown', e => { if (e.key === 'Escape') closeSheet(); });

/* ---------- search inspector ---------- */
function renderSearches() {
  const all = [];
  Object.entries(B.runs).forEach(([key, run]) => {
    const [tier, inst, cond] = key.split('::');
    run.searches.forEach(s => all.push({ tier, inst, cond, ...s }));
  });
  const host = $('#searches');
  const bar = el('div', 'bar');
  const sel = el('select');
  [['all', 'all searches'], ['empty', 'returned nothing'], ['exit2', 'usage errors (exit 2)'],
   ['unreadable', 'files walked but 0 chunks']].forEach(([v, t]) =>
    sel.append(new Option(t, v)));
  const cnt = el('span', 'count');
  bar.append(sel, cnt); host.append(bar);
  const wrap = el('div', 'panel scroll'), t = el('table');
  t.innerHTML = '<thead><tr>' + ['instance', 'arm', '#', 'command', 'exit', 'mode', 'files',
    'chunks', 'hits', 'ms'].map(h => `<th class="no">${h}</th>`).join('') + '</tr></thead>';
  const tb = el('tbody'); t.append(tb); wrap.append(t); host.append(wrap);
  const draw = () => {
    const f = sel.value;
    const rows = all.filter(s => {
      if (f === 'empty') return (s.stdout_bytes || 0) === 0 && s.exit !== 2;
      if (f === 'exit2') return s.exit === 2;
      if (f === 'unreadable') return s.trace && s.trace.files_walked > 0 && s.trace.n_chunks_considered === 0;
      return true;
    });
    cnt.textContent = `${rows.length} of ${all.length}`;
    tb.textContent = '';
    const frag = document.createDocumentFragment();
    rows.slice(0, 600).forEach(s => {
      const tr = el('tr');
      tr.append(el('td', 'mono', s.inst), el('td', 'mono ' + armClass(s.cond), s.cond),
                el('td', 'num', s.pos));
      const c = el('td', 'mono'); c.textContent = (s.argv || []).join(' ').slice(0, 90); tr.append(c);
      const ex = el('td', 'num');
      ex.append(el('span', 'pill ' + (s.exit === 2 ? 'bad' : s.exit === 1 ? 'warn' : 'good'), String(s.exit)));
      tr.append(ex);
      const tr2 = s.trace || {};
      tr.append(el('td', 'mono', tr2.mode || '—'), el('td', 'num', tr2.files_walked ?? '—'),
                el('td', 'num', tr2.n_chunks_considered ?? '—'), el('td', 'num', tr2.n_hits ?? '—'),
                el('td', 'num', s.wall_ms ?? '—'));
      frag.append(tr);
    });
    tb.append(frag);
    if (rows.length > 600) host.querySelector('.note')?.remove(),
      host.append(el('div', 'note', `showing the first 600 of ${rows.length}`));
  };
  sel.onchange = draw; draw();
}

/* ---------- boot ---------- */
renderGates(); renderScore(); renderSearches();
TIERS.forEach(t => $('#tier').append(new Option(t, t)));
['q', 'tier', 'outcome', 'metric'].forEach(id => {
  const n = $('#' + id); n.addEventListener(id === 'q' ? 'input' : 'change', renderTable);
});
document.querySelectorAll('#itable th[data-k]').forEach(th => {
  th.onclick = () => { const k = th.dataset.k;
    sortDir = (k === sortKey) ? -sortDir : 1; sortKey = k; renderTable(); };
});
renderTable();
"""


def build(bundle, out_path):
    bundle["scoreboard"] = scoreboard(bundle)
    prov = next(iter(bundle.get("provenance", {}).values()), {}) or {}
    payload = json.dumps(bundle, separators=(",", ":"))
    # `</` would end the script element early; escaping it is the one thing that
    # must not be forgotten when inlining JSON into HTML.
    payload = payload.replace("</", "<\\/")

    chips = "".join(
        f'<span class="chip">{html.escape(k)} <b>{html.escape(str(v)[:24])}</b></span>'
        for k, v in (
            ("model", prov.get("model")),
            ("semgrep", (prov.get("semgrep_sha256") or "")[:12]),
            ("claude", prov.get("claude_version")),
            ("dataset", (prov.get("dataset_sha256") or "")[:12]),
            ("captured", bundle.get("generated_at", "")[:16]),
        ) if v)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>semgrep loc-bench — results and trajectories</title>
<style>{CSS}</style></head>
<body>
<header><div class="hrow">
  <div><h1>loc-bench results &amp; trajectories</h1>
  <p class="sub">every search an agent ran, what came back, and what the engine did</p></div>
  <div class="chips">{chips}<button id="theme">theme</button></div>
</div></header>

<div class="wrap">
  <section><h2>gate</h2>
    <p class="sub">the verdict <code>triage.py</code> produced, not a second opinion</p>
    <div id="gates"></div></section>

  <section><h2>scoreboard</h2>
    <p class="sub">paired on instances where both arms completed</p>
    <div id="score"></div></section>

  <section><h2>tasks</h2>
    <p class="sub">one row per task; click to open both arms side by side</p>
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
      <span class="count" id="count"></span>
    </div>
    <div class="panel scroll"><table id="itable"><thead><tr>
      <th data-k="instance_id">instance</th><th data-k="tier">sample</th>
      <th data-k="category">category</th>
      <th data-k="outcome">rg</th><th data-k="outcome">desc-v5</th>
      <th data-k="searches" class="num">searches</th><th data-k="cost" class="num">cost</th>
      <th class="no"></th></tr></thead><tbody id="tbody"></tbody></table></div>
  </section>

  <section><h2>search inspector</h2>
    <p class="sub">every captured invocation — where a §16.11-class bug shows as a pattern</p>
    <div id="searches"></div></section>
</div>

<div class="overlay" id="overlay"></div>
<div class="sheet" id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle">
  <header><div class="hrow" style="padding:0">
    <div><h1 id="sheetTitle"></h1><p class="sub" id="sheetSub"></p></div>
    <div class="chips"><button id="sheetClose">close (esc)</button></div>
  </div></header>
  <div class="body" id="sheetBody"></div>
</div>

<script type="application/json" id="bundle">{payload}</script>
<script>{JS}</script>
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
