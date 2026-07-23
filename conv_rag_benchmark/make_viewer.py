"""
Build a self-contained interactive HTML viewer:
  * the typed knowledge graph (nodes coloured by type, edges labelled with the relation)
  * the generated questions, each with its query type, gold, source doc, the evidence
    context used, and the RELATIONS that connect entities in that evidence
  * the list of source docs

Open the resulting output/graph_viewer.html in any browser (needs internet for the
vis-network library from CDN).

Run: python -m conv_rag_benchmark.make_viewer
"""
from __future__ import annotations

import json
import os

from .config import Config
from .datasets.loader import DatasetLoader
from .embeddings import Embedder
from .graph.graph_builder import GraphBuilder
from .llm import LLM


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Conversational-RAG Benchmark Viewer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#0e1116;color:#e6edf3}
  h1{font-size:18px;margin:10px 14px}
  #wrap{display:flex;height:calc(100vh - 46px)}
  #graph{flex:2;border-right:1px solid #30363d;background:#0e1116}
  #side{flex:1;overflow:auto;padding:10px 14px;min-width:340px}
  .legend{margin:0 14px 6px;font-size:12px}
  .chip{display:inline-block;padding:2px 8px;border-radius:10px;margin:2px;font-size:11px;color:#0e1116;font-weight:600}
  .q{border:1px solid #30363d;border-radius:8px;padding:8px 10px;margin:8px 0;cursor:pointer;background:#161b22}
  .q:hover{border-color:#1c8fa8}
  .qtype{display:inline-block;background:#1c8fa8;color:#001;border-radius:6px;padding:1px 7px;font-size:11px;font-weight:700}
  .qtext{font-weight:600;margin:5px 0}
  .meta{font-size:12px;color:#9fb0bd;margin:3px 0}
  .rel{font-family:Consolas,monospace;font-size:11px;color:#8fd6c0}
  .gold{font-size:12px;color:#cfe6ee}
  .ev{font-size:11px;color:#7d8a96;border-left:2px solid #30363d;padding-left:6px;margin:3px 0}
  input{width:100%;padding:6px;border-radius:6px;border:1px solid #30363d;background:#0e1116;color:#e6edf3;margin-bottom:6px}
  .tabs button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 10px;margin-right:4px;cursor:pointer}
  .tabs button.on{background:#1c8fa8;color:#001;font-weight:700}
  .doc{font-size:12px;padding:4px 0;border-bottom:1px solid #21262d}
</style></head>
<body>
<h1>Conversational-RAG Benchmark — Graph &amp; Questions</h1>
<div class="legend" id="legend"></div>
<div id="wrap">
  <div id="graph"></div>
  <div id="side">
    <div class="tabs"><button id="tabQ" class="on" onclick="show('Q')">Questions</button>
      <button id="tabD" onclick="show('D')">Docs</button></div>
    <input id="search" placeholder="filter questions..." oninput="renderQ()">
    <div id="list"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const PALETTE = ["#1c8fa8","#2e8b6b","#e0922f","#c0492f","#6f6fd6","#0e4d64","#b05fa0","#5b7886","#8a8a3a"];
const types = [...new Set(DATA.nodes.map(n=>n.type))];
const colorOf = t => PALETTE[types.indexOf(t)%PALETTE.length];
document.getElementById('legend').innerHTML = types.map(t=>
  `<span class="chip" style="background:${colorOf(t)}">${t}</span>`).join('');

const nodes = new vis.DataSet(DATA.nodes.map(n=>({id:n.id,label:n.label,
  color:{background:colorOf(n.type),border:'#0e1116'},font:{color:'#e6edf3',size:12},title:n.type})));
const edges = new vis.DataSet(DATA.edges.map((e,i)=>({id:i,from:e.from,to:e.to,label:e.label,
  arrows:'to',font:{size:9,color:'#8fd6c0',strokeWidth:0},color:{color:'#3a4654'}})));
const net = new vis.Network(document.getElementById('graph'),{nodes,edges},{
  physics:{stabilization:true,barnesHut:{springLength:130,gravitationalConstant:-6000}},
  edges:{smooth:{type:'dynamic'}},nodes:{shape:'dot',size:12}});

function highlight(rels){
  const keepN=new Set(), keepE=new Set();
  rels.forEach(r=>{ keepN.add(r.source); keepN.add(r.target);
    edges.forEach(e=>{ const a=nodes.get(e.from), b=nodes.get(e.to);
      if(a&&b&&a.label===r.source&&b.label===r.target&&e.label===r.relation) keepE.add(e.id);});});
  nodes.update(DATA.nodes.map(n=>({id:n.id,color:{background:keepN.has(n.label)?colorOf(n.type):'#222a33',
    border:keepN.has(n.label)?'#fff':'#0e1116'},opacity:keepN.size?(keepN.has(n.label)?1:0.25):1})));
  edges.update(DATA.edges.map((e,i)=>({id:i,color:{color:keepE.has(i)?'#1c8fa8':'#283039'},
    width:keepE.has(i)?2.5:1})));
}

let mode='Q';
function show(m){mode=m;document.getElementById('tabQ').className=m==='Q'?'on':'';
  document.getElementById('tabD').className=m==='D'?'on':'';
  document.getElementById('search').style.display=m==='Q'?'block':'none'; render();}
function render(){ mode==='Q'?renderQ():renderD(); }
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function renderQ(){
  const f=(document.getElementById('search').value||'').toLowerCase();
  const html = DATA.questions.filter(q=>!f||q.question.toLowerCase().includes(f)).map((q,i)=>`
   <div class="q" onclick='hl(${JSON.stringify(q.relations_used)})'>
     <span class="qtype">${esc(q.type)}</span> <span class="meta">doc ${esc(''+q.source_doc)} · turn ${q.turn}</span>
     <div class="qtext">${esc(q.question)}</div>
     <div class="gold">✔ gold: ${esc(q.gold)}</div>
     <div class="meta">relations used (${q.relations_used.length}):</div>
     ${q.relations_used.map(r=>`<div class="rel">${esc(r.source)} —[${esc(r.relation)}]→ ${esc(r.target)}</div>`).join('')||'<div class="rel" style="color:#666">(none matched)</div>'}
     ${q.evidence.map(e=>`<div class="ev">${esc(e.slice(0,160))}…</div>`).join('')}
   </div>`).join('');
  document.getElementById('list').innerHTML=html;
}
function hl(rels){highlight(rels);}
function renderD(){
  document.getElementById('list').innerHTML = DATA.docs.map(d=>`<div class="doc">📄 doc ${esc(''+d)}</div>`).join('');
}
renderQ();
</script>
</body></html>
"""


LABELS = {"multihoprag": "MultiHopRAG", "medqa": "MedQA", "hotpotqa": "HotpotQA",
          "2wikimultihopqa": "2WikiMultiHopQA", "musique": "MuSiQue"}


def _dedup_edges(edges):
    agg = {}
    for e in edges:
        k = (e["source"], e["target"], e.get("relation"))
        if k in agg:
            agg[k]["weight"] += 1
        else:
            agg[k] = {"source": e["source"], "target": e["target"],
                      "relation": e.get("relation"), "weight": e.get("weight", 1)}
    return list(agg.values())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multihoprag")
    args = ap.parse_args(argv)
    ds_label = LABELS.get(args.dataset, args.dataset)

    config = Config.load(dataset=args.dataset, max_samples=50,
                         prefer_local_embeddings=False)
    config.ensure_dirs()
    out_dir = os.path.join(config.output_dir, ds_label)
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "graph.json")

    # use the graph the benchmark already saved if present, else build it
    if os.path.exists(graph_path):
        print(f"# loading saved graph from {graph_path}")
        g = json.load(open(graph_path, encoding="utf-8"))
        nodes, edges = g["nodes"], _dedup_edges(g["edges"])
    else:
        gen_llm = LLM(model=config.gen_model, config=config)
        embedder = Embedder(config=config, llm=gen_llm)
        seeds = DatasetLoader(args.dataset, max_samples=50).load()
        chunks = [c for s in seeds for c in s.context if c and c.strip()][:600]
        print(f"# building typed graph over {len(chunks)} chunks ...")
        kg = GraphBuilder(config=config, llm=gen_llm, embedder=embedder,
                          graph_mode="typed").build(chunks)
        nodes, edges = kg.nodes(), kg.edges()
        kg.save(graph_path)

    id2name = {n["id"]: (n.get("entity") or n["id"]) for n in nodes}

    # enrich the benchmark questions with the relations present in their evidence
    bench_path = os.path.join(out_dir, "benchmark_random.json")
    bench = json.load(open(bench_path, encoding="utf-8"))
    questions = []
    for c in bench["conversations"]:
        for t in c["turns"]:
            ev = " ".join(t.get("question_evidence_context", [])).lower()
            rels, seen = [], set()
            for e in edges:
                s = id2name.get(e["source"], e["source"])
                o = id2name.get(e["target"], e["target"])
                if s and o and s.lower() in ev and o.lower() in ev:
                    key = (s, e.get("relation"), o)
                    if key in seen:
                        continue
                    seen.add(key)
                    rels.append({"source": s, "target": o, "relation": e.get("relation")})
            questions.append({
                "conv": c["conversation_id"], "turn": t["turn_id"],
                "type": t["query_type"], "question": t["question"],
                "gold": t["gold_answer"], "source_doc": t.get("source_doc"),
                "evidence": t.get("question_evidence_context", [])[:3],
                "relations_used": rels[:8],
            })
    docs = sorted(set(str(t.get("source_doc"))
                      for c in bench["conversations"] for t in c["turns"]))

    data = {
        "nodes": [{"id": n["id"], "label": n.get("entity") or n["id"],
                   "type": n.get("type") or "Other"} for n in nodes],
        "edges": [{"from": e["source"], "to": e["target"],
                   "label": (e.get("relation") or "")
                            + (" x" + str(e["weight"]) if e.get("weight", 1) > 1 else "")}
                  for e in edges],
        "questions": questions, "docs": docs,
    }
    out = os.path.join(out_dir, "graph_viewer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"wrote {out}")
    print(f"  nodes {len(nodes)} | edges {len(edges)} | questions {len(questions)} | docs {len(docs)}")


if __name__ == "__main__":
    main()
