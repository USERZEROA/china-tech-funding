"""Small live page for the funding research loop."""

import json
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string

from src.agent_loop import AGENT_RUNS, graph_spec, run_agent_loop


app = Flask(__name__)
run_lock = threading.Lock()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Funding model loop</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #192638; background: #f4f7fb; }
    header { background: #17324f; color: white; padding: 22px 34px; }
    header h1 { margin: 0 0 7px; font-size: 25px; }
    header p { margin: 0; color: #d7e4f1; }
    main { max-width: 1500px; margin: 24px auto; padding: 0 22px; }
    .card { background: white; border: 1px solid #dce5ee; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 7px #15233c0d; }
    .row { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
    h2 { font-size: 20px; margin: 0 0 14px; }
    button { background: #1766a3; color: white; border: 0; border-radius: 6px; padding: 10px 16px; font-weight: bold; cursor: pointer; }
    button:disabled { opacity: .55; cursor: default; }
    .small { color: #53657a; font-size: 14px; }
    .graph-wrap { overflow-x: auto; }
    svg { width: 100%; min-width: 1380px; height: 390px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 9px 11px; border-bottom: 1px solid #e3e9f0; text-align: left; }
    th { background: #f1f5f9; }
    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .metric { font-size: 22px; font-weight: bold; color: #164d75; }
    @media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>China tech funding model loop</h1><p>Company-grouped training, validation feedback, and one final test</p></header>
  <main>
    <section class="card">
      <div class="row"><div><h2>Live agent graph</h2><div id="run-status" class="small">Loading run status...</div></div><button id="run-button">Run the loop</button></div>
      <div class="graph-wrap"><svg id="graph" viewBox="0 0 1390 390" aria-label="Actual agent workflow"></svg></div>
      <div id="source" class="small"></div>
    </section>
    <div class="cols">
      <section class="card"><h2>Validation experiments</h2><div id="experiments" class="small">No completed run yet.</div></section>
      <section class="card"><h2>Final test</h2><div id="test" class="small">The test set opens after validation selection.</div></section>
    </div>
    <section class="card"><h2>Execution events</h2><div id="events" class="small">Waiting for a run.</div></section>
  </main>
  <script>
    const NS = 'http://www.w3.org/2000/svg';
    const el = (name, attrs = {}) => { const node = document.createElementNS(NS, name); for (const [k,v] of Object.entries(attrs)) node.setAttribute(k, v); return node; };
    const fmt = x => typeof x === 'number' ? x.toFixed(3) : '—';
    const matrix = m => m ? `[${m[0].join(', ')}] / [${m[1].join(', ')}]` : '—';

    function drawGraph(spec, events) {
      const svg = document.getElementById('graph'); svg.replaceChildren();
      const defs = el('defs'); const marker = el('marker', { id:'arrow', markerWidth:10, markerHeight:10, refX:8, refY:3, orient:'auto' });
      marker.append(el('path', { d:'M0,0 L0,6 L9,3 z', fill:'#8da5bb' })); defs.append(marker); svg.append(defs);
      const nodes = Object.fromEntries(spec.nodes.map(n => [n.id, n]));
      for (const edge of spec.edges) {
        const a=nodes[edge.source], b=nodes[edge.target];
        const d=edge.route==='revise' ? `M ${a.x+95} ${a.y+88} V 325 H ${b.x+95} V ${b.y+88}` : `M ${a.x+190} ${a.y+44} H ${b.x}`;
        svg.append(el('path', { d, fill:'none', stroke:'#8da5bb', 'stroke-width':'2.5', 'marker-end':'url(#arrow)' }));
        if (edge.route==='revise') {
          const label=el('text', { x:640, y:344, 'text-anchor':'middle', fill:'#53657a', 'font-size':14 }); label.textContent='revise from validation feedback'; svg.append(label);
        }
      }
      const latest = {};
      for (const item of events) latest[item.node]=item.phase;
      for (const node of spec.nodes) {
        const phase=latest[node.id]; const fill=phase==='completed'?'#e5f5e9':phase==='started'?'#fff1ce':phase==='failed'?'#ffe6e6':'#edf3f9';
        const stroke=phase==='completed'?'#23834a':phase==='started'?'#cc9200':phase==='failed'?'#b92525':'#9eb3c8';
        svg.append(el('rect', { x:node.x, y:node.y, width:190, height:88, rx:10, fill, stroke, 'stroke-width':2 }));
        const title=el('text', { x:node.x+12, y:node.y+31, fill:'#192638', 'font-size':17, 'font-weight':'bold' }); title.textContent=node.title; svg.append(title);
        const role=el('text', { x:node.x+12, y:node.y+58, fill:'#52667b', 'font-size':11 }); role.textContent=node.role; svg.append(role);
        const status=el('text', { x:node.x+12, y:node.y+76, fill:stroke, 'font-size':11 }); status.textContent=phase||'pending'; svg.append(status);
      }
    }

    async function refresh() {
      const data=await (await fetch('/api/latest')).json();
      drawGraph(data.graph, data.events);
      const status=data.status||{};
      document.getElementById('run-status').textContent=`Run: ${status.run_id||'none'} · ${status.status||'idle'}`;
      document.getElementById('run-button').disabled=status.status==='running';
      const summary=data.summary;
      document.getElementById('source').textContent=summary ? `Source DAG run: ${summary.source_run_id} · Eligible: ${summary.eligible.rows} · Company overlap: ${summary.company_group_overlap}` : '';
      if (summary) {
        document.getElementById('experiments').innerHTML='<table><tr><th>Cycle</th><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>AP</th><th>Selected</th></tr>'+
          summary.experiments.map(x=>`<tr><td>${x.cycle}</td><td>${x.spec.family}</td><td>${fmt(x.validation.f1)}</td><td>${fmt(x.validation.precision)}</td><td>${fmt(x.validation.recall)}</td><td>${fmt(x.validation.average_precision)}</td><td>${x.cycle===summary.best_cycle?'Yes':''}</td></tr>`).join('')+'</table>';
        const m=summary.test_metrics;
        document.getElementById('test').innerHTML=`<div class="metric">F1 ${fmt(m.f1)}</div><p>Precision ${fmt(m.precision)} · Recall ${fmt(m.recall)} · AP ${fmt(m.average_precision)}</p><p>Confusion matrix [TN, FP] / [FN, TP]: ${matrix(m.confusion_matrix)}</p><p>Test rows: ${m.rows}; positive: ${m.positive}. Stop: ${summary.stop_reason}.</p>`;
      }
      document.getElementById('events').innerHTML=data.events.length ? '<table><tr><th>UTC time</th><th>Agent</th><th>Phase</th><th>Route</th></tr>'+
        data.events.map(x=>`<tr><td>${x.at_utc}</td><td>${x.node}</td><td>${x.phase}</td><td>${x.details?.route||''}</td></tr>`).join('')+'</table>' : 'Waiting for a run.';
    }
    document.getElementById('run-button').onclick=async()=>{ const response=await fetch('/run',{method:'POST'}); if(!response.ok) alert('A run is already in progress.'); await refresh(); };
    refresh(); setInterval(refresh, 1000);
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/latest")
def latest():
    runs = sorted(folder for folder in AGENT_RUNS.glob("*") if folder.is_dir())
    if not runs:
        return jsonify({"graph": graph_spec(), "events": [], "status": {"status": "idle"}, "summary": None})
    folder = runs[-1]
    status_file = folder / "status.json"
    summary_file = folder / "run_summary.json"
    events_file = folder / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()] if events_file.exists() else []
    return jsonify({"graph": graph_spec(), "events": events,
                    "status": json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {"status": "starting"},
                    "summary": json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else None})


@app.post("/run")
def start_run():
    if not run_lock.acquire(blocking=False):
        return jsonify({"error": "run already in progress"}), 409
    run_id = "agent__" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    def worker():
        try:
            run_agent_loop(run_id)
        finally:
            run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"run_id": run_id}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False, threaded=True)
