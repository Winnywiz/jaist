"""
make_quality_viewer.py — generate a self-contained HTML page to browse the generated
conversations by dataset / RAG / conversation, with per-turn quality flags.

Reads every benchmark run under result/benchmark_quality/, named
``{rag}_{dataset}_t{turns}_c{convos}.json``, computes weakness flags (repeated
answers, guard give-ups) per turn, and writes one HTML file with all the data
embedded. No server, no JSON reading needed.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

OUT = "result/quality_viewer.html"
_TOK = re.compile(r"[a-z0-9]+")


def _toks(s):
    return set(_TOK.findall((s or "").lower()))


def _is_repeat(gold, prev_golds, thresh=0.8):
    g = (gold or "").strip().lower()
    gt = _toks(g)
    if not g or not gt:
        return False
    for p in prev_golds:
        pt = _toks(p)
        if not pt:
            continue
        if g == p.lower().strip() or g in p.lower():
            return True
        if len(gt & pt) / len(gt | pt) >= thresh:
            return True
    return False


def build_data():
    datasets = {}
    # run files are named {rag}_{dataset}_t{turns}_c{convos}[_flags].json
    for p in sorted(glob.glob("result/benchmark_quality/*/*_t*_c*.json")):
        base = os.path.basename(p)
        # _summary files hold mean/std across repeats, not conversations
        if any(x in base for x in ("_docsim", "_doc_similarity", "_export",
                                   "_retrieval_alignment", "_summary")):
            continue
        ds = os.path.basename(os.path.dirname(p))
        d = json.load(open(p, encoding="utf-8"))
        rag = d.get("rag", "vector")
        # a dataset+RAG can have several runs (different turns/convos, e.g.
        # t10_c5 vs t20_c5) and several REPEATS of the same config (_r1, _r2, ...).
        # Key them all separately or the later file silently overwrites the earlier one.
        _m = re.search(r"_(t\d+_c\d+)", base)
        if _m:
            rag = f"{rag} · {_m.group(1)}"
        _r = re.search(r"_r(\d+)\.json$", base)
        if _r:
            rag = f"{rag} · run{_r.group(1)}"
        q = (d.get("quality", {}) or {}).get("E", {}) or {}
        by_type = {t: m.get("well_formed") for t, m in (d.get("e_by_query_type") or {}).items()}
        convos = []
        for c in d.get("conversations", []):
            prev_golds, turns, dups, giveups = [], [], 0, 0
            oc = Counter()
            for t in c.get("turns", []):
                dup = _is_repeat(t.get("gold", ""), prev_golds)
                dups += dup
                giveups += bool(t.get("guard_gave_up"))
                oc[t.get("outcome")] += 1
                turns.append({
                    "type": t.get("query_type"), "q": t.get("question", ""),
                    "gold": t.get("gold", ""), "rag": t.get("rag_answer", ""),
                    "outcome": t.get("outcome"), "dup": dup,
                    "gaveup": bool(t.get("guard_gave_up")),
                })
                prev_golds.append(t.get("gold", ""))
            convos.append({"turns": turns, "dups": dups, "giveups": giveups,
                           "outcomes": dict(oc)})
        datasets.setdefault(ds, {})[rag] = {
            "quality": {k: q.get(k) for k in ("well_formed", "gold_supported", "gold_correct")},
            "by_type": by_type,
            "n_conv": len(convos), "n_turns": sum(len(c["turns"]) for c in convos),
            "dup_turns": sum(c["dups"] for c in convos),
            "giveup_turns": sum(c["giveups"] for c in convos),
            "conversations": convos,
        }
    return datasets


HTML = r"""<title>Conversation Quality Inspector</title>
<style>
:root{
  --bg:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5; --ink:#182230; --muted:#5c6774;
  --border:#e1e6ec; --accent:#0e7a75; --accent-soft:#d6ebe9;
  --ok:#1f8a54; --ok-bg:#dcefe3; --warn:#b7791f; --warn-bg:#f6ecd6;
  --bad:#c23b3b; --bad-bg:#f7dede; --rep:#c15c00; --rep-bg:#f7e5d2;
  --shadow:0 1px 2px rgba(20,30,45,.06),0 4px 14px rgba(20,30,45,.05);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f141a; --surface:#161d25; --surface-2:#1d2630; --ink:#e6ebf1; --muted:#98a3b2;
  --border:#27303b; --accent:#3fb8b0; --accent-soft:#123634;
  --ok:#4bbd80; --ok-bg:#12281d; --warn:#d6a13f; --warn-bg:#2b2413; --bad:#e06666;
  --bad-bg:#2c1717; --rep:#e0842f; --rep-bg:#2c1e10; --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 18px rgba(0,0,0,.28);
}}
:root[data-theme="dark"]{
  --bg:#0f141a; --surface:#161d25; --surface-2:#1d2630; --ink:#e6ebf1; --muted:#98a3b2;
  --border:#27303b; --accent:#3fb8b0; --accent-soft:#123634;
  --ok:#4bbd80; --ok-bg:#12281d; --warn:#d6a13f; --warn-bg:#2b2413; --bad:#e06666;
  --bad-bg:#2c1717; --rep:#e0842f; --rep-bg:#2c1e10;
}
:root[data-theme="light"]{
  --bg:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5; --ink:#182230; --muted:#5c6774;
  --border:#e1e6ec; --accent:#0e7a75; --accent-soft:#d6ebe9;
  --ok:#1f8a54; --ok-bg:#dcefe3; --warn:#b7791f; --warn-bg:#f6ecd6; --bad:#c23b3b;
  --bad-bg:#f7dede; --rep:#c15c00; --rep-bg:#f7e5d2;
}
*{box-sizing:border-box}
body{margin:0}
.wrap{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);
  background:var(--bg);min-height:100vh;line-height:1.5;
  font-size:15px;-webkit-font-smoothing:antialiased}
.serif{font-family:Georgia,"Iowan Old Style",Cambria,"Times New Roman",serif}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.container{max-width:1080px;margin:0 auto;padding:0 20px}

header.top{border-bottom:1px solid var(--border);background:var(--surface);
  position:sticky;top:0;z-index:10;box-shadow:var(--shadow)}
.hbar{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;padding:16px 0 6px}
.hbar h1{font-size:19px;margin:0;letter-spacing:-.01em}
.hbar .sub{color:var(--muted);font-size:13px}
.controls{display:flex;gap:16px;flex-wrap:wrap;padding:8px 0 16px;align-items:flex-end}
.ctrl{display:flex;flex-direction:column;gap:4px}
.ctrl label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600}
select{font:inherit;font-size:14px;color:var(--ink);background:var(--surface);
  border:1px solid var(--border);border-radius:8px;padding:8px 10px;min-width:150px;cursor:pointer}
select:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

.strip{display:flex;gap:10px;flex-wrap:wrap;padding:18px 0 4px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:10px 14px;min-width:118px;box-shadow:var(--shadow)}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
.tile .v{font-size:22px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .v small{font-size:12px;color:var(--muted);font-weight:500}
.tile.warnflag .v{color:var(--rep)}

.convmeta{color:var(--muted);font-size:13px;padding:14px 0 2px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;
  padding:2px 9px;border-radius:999px;white-space:nowrap}
.chip.rep{background:var(--rep-bg);color:var(--rep)}
.chip.give{background:var(--warn-bg);color:var(--warn)}

.turns{display:flex;flex-direction:column;gap:14px;padding:12px 0 40px}
.turn{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:16px 18px;box-shadow:var(--shadow);position:relative}
.turn.dup{border-left:3px solid var(--rep)}
.turn .thead{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.tnum{font-size:12px;color:var(--muted);font-weight:600}
.badge{font-size:11px;font-weight:600;padding:2px 9px;border-radius:6px;letter-spacing:.02em;
  background:var(--surface-2);color:var(--muted);border:1px solid var(--border)}
.pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;text-transform:capitalize}
.pill.correct{background:var(--ok-bg);color:var(--ok)}
.pill.wrong,.pill.hallucinated{background:var(--bad-bg);color:var(--bad)}
.pill.abstained{background:var(--warn-bg);color:var(--warn)}
.flag{margin-left:auto;font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;
  background:var(--rep-bg);color:var(--rep)}
.q{font-size:18px;line-height:1.45;margin:2px 0 12px;text-wrap:pretty}
.field{display:grid;grid-template-columns:70px 1fr;gap:10px;padding:6px 0;border-top:1px solid var(--border)}
.field .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  font-weight:600;padding-top:2px}
.field .val{font-size:15px;line-height:1.5}
.field .val.gold{color:var(--ink)}
.field .val.rag{color:var(--muted)}

.weak{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:20px 22px;margin:8px 0 48px;box-shadow:var(--shadow)}
.weak h2{font-size:16px;margin:0 0 4px}
.weak p.lead{color:var(--muted);font-size:13px;margin:0 0 16px}
.weak ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:14px}
.weak li{display:grid;grid-template-columns:26px 1fr;gap:12px}
.weak li .n{font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}
.weak li b{font-weight:600}
.weak li .d{color:var(--muted);font-size:14px;margin-top:2px}
.bytype{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.bytype .t{font-size:12px;padding:3px 8px;border-radius:6px;background:var(--surface-2);
  border:1px solid var(--border);font-variant-numeric:tabular-nums}
.bytype .t.low{background:var(--rep-bg);color:var(--rep);border-color:transparent}
.foot{color:var(--muted);font-size:12px;padding:0 0 40px;text-align:center}
</style>

<div class="wrap">
<header class="top"><div class="container">
  <div class="hbar">
    <h1>Conversation Quality Inspector</h1>
    <span class="sub">browse generated questions, golds &amp; RAG answers by dataset and system</span>
  </div>
  <div class="controls">
    <div class="ctrl"><label for="ds">Dataset</label><select id="ds"></select></div>
    <div class="ctrl"><label for="rag">RAG under test</label><select id="rag"></select></div>
    <div class="ctrl"><label for="cv">Conversation</label><select id="cv"></select></div>
  </div>
</div></header>

<div class="container">
  <div class="strip" id="strip"></div>
  <div class="convmeta" id="convmeta"></div>
  <div class="turns" id="turns"></div>
  <div class="weak" id="weak"></div>
  <div class="foot">Generated from result/benchmark_quality/ &middot; 20 conversations &times; 6 turns per dataset.</div>
</div>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const dsSel=$("#ds"), ragSel=$("#rag"), cvSel=$("#cv");

function opt(v,t){const o=document.createElement("option");o.value=v;o.textContent=t;return o;}
function pct(x){return x==null?"&ndash;":Math.round(x*100)+"%";}

function fillDatasets(){
  Object.keys(DATA).forEach(d=>dsSel.appendChild(opt(d,d)));
  fillRags();
}
function fillRags(){
  ragSel.innerHTML="";
  Object.keys(DATA[dsSel.value]).forEach(r=>ragSel.appendChild(opt(r,r)));
  fillConvs();
}
function fillConvs(){
  cvSel.innerHTML="";
  const rec=DATA[dsSel.value][ragSel.value];
  rec.conversations.forEach((c,i)=>{
    const flags=[]; if(c.dups)flags.push(c.dups+" rep"); if(c.giveups)flags.push(c.giveups+" give-up");
    cvSel.appendChild(opt(i,"Conversation "+(i+1)+(flags.length?"  ("+flags.join(", ")+")":"")));
  });
  render();
}

function render(){
  const rec=DATA[dsSel.value][ragSel.value];
  const q=rec.quality;
  // summary tiles
  $("#strip").innerHTML=`
    ${tile("gold_supported",pct(q.gold_supported))}
    ${tile("gold_correct",pct(q.gold_correct))}
    ${tile("well_formed",pct(q.well_formed))}
    ${tile("conversations",rec.n_conv+" &middot; "+rec.n_turns+" turns")}
    ${tile("repeated answers",rec.dup_turns+"<small> /"+rec.n_turns+"</small>",rec.dup_turns>0)}
  `;
  const c=rec.conversations[+cvSel.value];
  const om=Object.entries(c.outcomes).map(([k,v])=>`${v} ${k}`).join(" &middot; ");
  $("#convmeta").innerHTML=`<span><b>Conversation ${+cvSel.value+1}</b></span><span>${om}</span>`+
    (c.dups?`<span class="chip rep">&#9888; ${c.dups} repeated answer${c.dups>1?"s":""}</span>`:"")+
    (c.giveups?`<span class="chip give">${c.giveups} guard give-up</span>`:"");
  $("#turns").innerHTML=c.turns.map((t,i)=>turnCard(t,i)).join("");
  renderWeak(rec);
}
function tile(k,v,warn){return `<div class="tile ${warn?'warnflag':''}"><div class="k">${k}</div><div class="v">${v}</div></div>`;}
function esc(s){return (s||"").replace(/[&<>]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));}
function turnCard(t,i){
  const oc=(t.outcome||"").toLowerCase();
  return `<div class="turn ${t.dup?'dup':''}">
    <div class="thead">
      <span class="tnum">Turn ${i+1}</span>
      <span class="badge">${esc(t.type)}</span>
      <span class="pill ${oc}">${esc(t.outcome)}</span>
      ${t.dup?'<span class="flag">&#9888; repeats an earlier answer</span>':''}
      ${t.gaveup?'<span class="flag">guard gave up</span>':''}
    </div>
    <div class="q serif">${esc(t.q)}</div>
    <div class="field"><div class="lbl">Gold</div><div class="val gold serif">${esc(t.gold)}</div></div>
    <div class="field"><div class="lbl">RAG</div><div class="val rag serif">${esc(t.rag)}</div></div>
  </div>`;
}
function renderWeak(rec){
  const dupRate=rec.n_turns?Math.round(100*rec.dup_turns/rec.n_turns):0;
  const giveRate=rec.n_turns?Math.round(100*rec.giveup_turns/rec.n_turns):0;
  const bt=Object.entries(rec.by_type).sort((a,b)=>(a[1]??9)-(b[1]??9));
  const btHtml=bt.map(([t,v])=>`<span class="t ${v!=null&&v<0.5?'low':''}">${t}: ${pct(v)}</span>`).join("");
  $("#weak").innerHTML=`
    <h2>Known weaknesses (for this dataset / RAG)</h2>
    <p class="lead">Computed live from the turns above, plus the standing limitations of the method.</p>
    <ul>
      <li><span class="n">1</span><div><b>Repetitive answers.</b> ${rec.dup_turns} of ${rec.n_turns} turns (${dupRate}%) restate a fact already given earlier in their conversation. The no-repeat guard cuts most, but coreference/correction turns sometimes still circle back.</div></li>
      <li><span class="n">2</span><div><b>well_formed penalises coreference by design.</b> Pronoun and abstention question types score low because they are deliberately not self-contained &mdash; the metric mistakes that for poor phrasing. Per-type well_formed:<div class="bytype">${btHtml}</div></div></li>
      <li><span class="n">3</span><div><b>Occasional gold drift.</b> Some Topic-Shift / Multi-Hop golds echo the previous turn's answer instead of answering the new question &mdash; watch the turns flagged wrong.</div></li>
      <li><span class="n">4</span><div><b>Guard give-ups.</b> ${rec.giveup_turns} turns (${giveRate}%) failed verification 3&times; and were kept anyway; their answer key is less trustworthy.</div></li>
      <li><span class="n">5</span><div><b>LLM judges LLM.</b> Questions, golds and quality scores all come from the same model family &mdash; trust relative comparisons more than absolute values.</div></li>
    </ul>`;
}
dsSel.addEventListener("change",fillRags);
ragSel.addEventListener("change",fillConvs);
cvSel.addEventListener("change",render);
fillDatasets();
</script>
"""


def main():
    data = build_data()
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    nds = len(data)
    nrag = sum(len(v) for v in data.values())
    print(f"wrote {OUT} | {nds} datasets, {nrag} dataset-RAG combos")


if __name__ == "__main__":
    main()
