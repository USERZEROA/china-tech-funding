"""Read-only dashboard for the Bonus workflow evidence."""

import json

from flask import Blueprint, jsonify, render_template_string

from src import bonus_common


bonus_page = Blueprint("bonus_page", __name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Funding workflow comparison</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f7fb;color:#1c3047;margin:0}header{background:#17324f;color:white;padding:24px 32px}h1{margin:0 0 6px;font-size:26px}header p{margin:0;color:#dce7f1}main{max-width:1500px;margin:24px auto;padding:0 20px}.card{background:white;border:1px solid #dce5ee;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 2px 7px #15233c0d}h2{margin:0 0 13px;font-size:20px}.muted{color:#53657a}.flow{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:14px 0}.step{min-width:112px;max-width:125px;box-sizing:border-box;border:2px solid #9eb3c8;border-radius:9px;background:#edf3f9;padding:12px}.step.done{background:#e5f5e9;border-color:#23834a}.step.route{background:#fff1ce;border-color:#c58d1b}.step strong{display:block;margin-bottom:7px}.step small{color:#53657a}.arrow{align-self:center;font-size:26px;color:#7794aa}.tag{display:inline-block;background:#e7eef5;border-radius:5px;padding:5px 8px;margin:3px 5px 3px 0;font-size:13px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e0e7ee}th{background:#f1f5f9}.scroll{overflow-x:auto}.pass{color:#187341;font-weight:bold}.fail{color:#b22222;font-weight:bold}.metrics{font-size:24px;color:#164d75;font-weight:bold}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:850px){.split{grid-template-columns:1fr}}
</style></head><body><header><h1>China tech funding workflow comparison</h1><p>Fixed Airflow, classroom agent loop, and adaptive Planner graph</p></header><main>
<section class="card"><h2>Adaptive execution graph</h2><div id="status" class="muted">Loading real run evidence...</div><div id="flow" class="flow"></div><div id="plan"></div><p id="route" class="muted"></p></section>
<section class="card"><h2>Three workflow comparison</h2><div id="comparison" class="scroll"></div><p id="finding" class="muted"></p></section>
<div class="split"><section class="card"><h2>Evaluation suite</h2><div id="cases" class="scroll"></div></section><section class="card"><h2>Final holdout</h2><div id="holdout"></div></section></div>
<section class="card"><h2>Adaptive execution events</h2><div id="events" class="scroll"></div></section>
</main><script>
const fmt=x=>typeof x==='number'?x.toFixed(3):'—';
function table(headers, rows){const t=document.createElement('table'),head=document.createElement('tr');headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.append(th)});t.append(head);rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(value=>{const td=document.createElement('td');td.textContent=typeof value==='object'?JSON.stringify(value):String(value);tr.append(td)});t.append(tr)});return t}
function title(node){return node.replaceAll('_',' ').replace(/\\b\\w/g,c=>c.toUpperCase())}
function branchText(action){const s=action.model_spec;if(!s)return 'final test';return s.family==='logistic_regression'?`${action.target}: LR C=${s.C}, weight=${s.class_weight||'none'}`:`${action.target}: RF depth=${s.max_depth}, leaf=${s.min_samples_leaf}, weight=${s.class_weight||'none'}`}
async function load(){const data=await(await fetch('/api/bonus/latest')).json();if(data.error){document.getElementById('status').textContent=data.error;return}
const comparison=data.comparison, evaluation=data.evaluation, adaptive=data.adaptive, plan=data.planner.plan;
document.getElementById('status').textContent=`Run ${adaptive.run_id} · Planner: ${data.planner.origin} (${data.planner.model||'local rules'}) · ${evaluation.passed}/${evaluation.total} evaluation cases passed`;
const sequence=['data_steward','planner_agent'];adaptive.events.forEach(e=>{if(e.phase==='completed'&&['linear_agent','forest_agent','quality_agent'].includes(e.node))sequence.push(e.node);if(e.phase==='routed')sequence.push('judge_agent')});sequence.push('final_test');
const flow=document.getElementById('flow');sequence.forEach((name,i)=>{if(i){const arrow=document.createElement('span');arrow.className='arrow';arrow.textContent='→';flow.append(arrow)}const box=document.createElement('div');box.className='step done'+(name==='judge_agent'?' route':'');const strong=document.createElement('strong');strong.textContent=title(name);const small=document.createElement('small');small.textContent=name==='planner_agent'?'Graph plan':name==='judge_agent'?'Validation route':name==='final_test'?'Test once':'Executed';box.append(strong,small);flow.append(box)});
const planBox=document.getElementById('plan');[['First model',plan.first_agent],['Low precision action',branchText(data.graph.branch_actions.low_precision)],['Low recall action',branchText(data.graph.branch_actions.low_recall)],['Quality pass',branchText(data.graph.branch_actions.quality_pass)],['Precision gate',plan.precision_gate],['Recall gate',plan.recall_gate],['Retry limit',plan.retry_limit]].forEach(([k,v])=>{const tag=document.createElement('span');tag.className='tag';tag.textContent=`${k}: ${v}`;planBox.append(tag)});
document.getElementById('route').textContent='Actual routes: '+adaptive.routes.map(x=>`${x.focus} → ${x.target}`).join(' | ')+`. Stop: ${adaptive.stop_reason}.`;
const rows=comparison.rows.map(x=>[x.workflow,x.run_id,x.experiments,fmt(x.validation_f1),fmt(x.test_f1),fmt(x.test_precision),fmt(x.test_recall),fmt(x.test_ap),x.retry_events,x.stop_reason]);document.getElementById('comparison').append(table(['Workflow','Run ID','Fits','Val F1','Test F1','Precision','Recall','AP','Retries','Stop'],rows));
document.getElementById('finding').textContent=comparison.summary;
const cases=evaluation.cases.map(c=>[c.id,`${c.score}/${c.pass_threshold}`,c.passed?'PASS':'FAIL',c.expected]);document.getElementById('cases').append(table(['Case','Score / threshold','Result','Expected'],cases));
const m=adaptive.test_metrics;document.getElementById('holdout').innerHTML=`<div class="metrics">F1 ${fmt(m.f1)} · AP ${fmt(m.average_precision)}</div><p>Precision ${fmt(m.precision)} · Recall ${fmt(m.recall)}</p><p>Confusion matrix [TN, FP] / [FN, TP]: [${m.confusion_matrix[0].join(', ')}] / [${m.confusion_matrix[1].join(', ')}]</p><p>Rows ${m.rows}, positives ${m.positive}. Selection used validation only.</p>`;
document.getElementById('events').append(table(['UTC','Node','Phase','Route / cycle'],adaptive.events.map(e=>[e.at_utc,e.node,e.phase,e.details?.target||e.details?.cycle||''])));
}load();
</script></body></html>"""


@bonus_page.get("/bonus")
def bonus():
    return render_template_string(PAGE)


@bonus_page.get("/api/bonus/latest")
def latest_bonus():
    folders = sorted(path for path in bonus_common.ROOT.glob("comparison__*/workflow_comparison.json"))
    if not folders:
        return jsonify({"error": "Run the three workflows and evaluation suite first."}), 404
    folder = folders[-1].parent
    comparison = json.loads((folder / "workflow_comparison.json").read_text(encoding="utf-8"))
    evaluation = json.loads((folder / "evaluation_results.json").read_text(encoding="utf-8"))
    adaptive_id = evaluation["run_ids"]["adaptive_agent"]
    adaptive_folder = bonus_common.folder_for(adaptive_id)
    summary = json.loads((adaptive_folder / "run_summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (adaptive_folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    planner = json.loads((adaptive_folder / "planner_result.json").read_text(encoding="utf-8"))
    graph = json.loads((adaptive_folder / "graph_plan.json").read_text(encoding="utf-8"))
    return jsonify({"comparison": comparison, "evaluation": evaluation,
                    "adaptive": {**summary, "events": events}, "planner": planner, "graph": graph})
