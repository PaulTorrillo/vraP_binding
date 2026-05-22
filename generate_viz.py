#!/usr/bin/env python3
"""
Generate a self-contained HTML visualization from an AlphaFold Server output zip.
Usage: python generate_viz.py [zip_path] [output.html]
Defaults: fold_run_4_long.zip -> visualization.html
"""

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_cif_residue_plddts(cif_text: str):
    """Return (chain_A_plddts, chain_B_plddts) averaged per residue from B-factors."""
    residue_atoms: dict = defaultdict(list)
    for line in cif_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            chain = parts[6]       # label_asym_id
            resnum = int(parts[8]) # label_seq_id
            bfac = float(parts[14])# B_iso_or_equiv (pLDDT)
            residue_atoms[(chain, resnum)].append(bfac)
        except (ValueError, IndexError):
            continue

    def chain_sorted(c):
        return [round(sum(v) / len(v), 2)
                for _, v in sorted(
                    ((r, vals) for (ch, r), vals in residue_atoms.items() if ch == c),
                    key=lambda x: x[0]
                )]

    return chain_sorted("A"), chain_sorted("B")


def load_models(zip_path: str):
    models = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Detect job name prefix
        cif_files = [n for n in names if "_model_" in n and n.endswith(".cif")]
        prefix = cif_files[0].rsplit("_model_", 1)[0]

        indices = sorted(int(n.split("_model_")[1].split(".")[0]) for n in cif_files)

        req = json.loads(zf.read(next(n for n in names if n.endswith("_job_request.json"))))
        if isinstance(req, list):
            req = req[0]

        seqs = []
        for s in req.get("sequences", []):
            pc = s.get("proteinChain", {})
            if pc.get("sequence"):
                seqs.append(pc["sequence"])

        for i in indices:
            cif = zf.read(f"{prefix}_model_{i}.cif").decode()
            summary = json.loads(zf.read(f"{prefix}_summary_confidences_{i}.json"))
            full = json.loads(zf.read(f"{prefix}_full_data_{i}.json"))
            plddt_a, plddt_b = parse_cif_residue_plddts(cif)
            models.append({
                "index": i,
                "cif": cif,
                "ranking_score": round(summary["ranking_score"], 3),
                "iptm": round(summary.get("iptm", 0), 3),
                "ptm": round(summary.get("ptm", 0), 3),
                "chain_pair_iptm": summary.get("chain_pair_iptm", []),
                "pae": full["pae"],
                "plddt_a": plddt_a,
                "plddt_b": plddt_b,
                "token_chain_ids": full["token_chain_ids"],
            })

    models.sort(key=lambda m: -m["ranking_score"])

    chain_a_len = sum(1 for c in models[0]["token_chain_ids"] if c == "A")
    chain_b_len = sum(1 for c in models[0]["token_chain_ids"] if c == "B")

    job_name = req.get("name", prefix)
    seq_a = seqs[0] if len(seqs) > 0 else ""
    seq_b = seqs[1] if len(seqs) > 1 else ""

    return models, job_name, chain_a_len, chain_b_len, seq_a, seq_b


def generate_html(models, job_name, chain_a_len, chain_b_len, seq_a, seq_b):
    data_js = json.dumps(models, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaFold — {job_name}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{padding:.75rem 1.25rem;border-bottom:1px solid #30363d;flex-shrink:0}}
header h1{{font-size:1.1rem;font-weight:600}}
header p{{font-size:.8rem;color:#8b949e;margin-top:.2rem}}
.tabs{{display:flex;gap:.4rem;padding:.5rem 1.25rem;border-bottom:1px solid #30363d;background:#161b22;flex-shrink:0;align-items:center}}
.tabs-label{{font-size:.75rem;color:#8b949e;margin-right:.3rem}}
.tab{{padding:.3rem .75rem;border-radius:5px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.78rem;transition:background .15s}}
.tab.active{{background:#1f6feb;border-color:#388bfd;color:#fff}}
.tab:hover:not(.active){{background:#30363d;color:#e6edf3}}
.stats{{display:flex;gap:1.5rem;padding:.45rem 1.25rem;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0;flex-wrap:wrap}}
.stat{{display:flex;flex-direction:column}}
.stat-label{{font-size:.65rem;text-transform:uppercase;letter-spacing:.05em;color:#8b949e}}
.stat-value{{font-size:.9rem;font-weight:600;color:#e6edf3}}
.main{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:1px;background:#30363d;flex:1;min-height:0}}
.panel{{background:#0d1117;padding:.65rem;display:flex;flex-direction:column;min-height:0}}
.panel-title{{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin-bottom:.4rem;flex-shrink:0}}
#viewer-wrap{{grid-row:1/3;display:flex;flex-direction:column;background:#0d1117;padding:.65rem;min-height:0}}
.color-btns{{display:flex;gap:.35rem;flex-wrap:wrap;flex-shrink:0;margin-bottom:.4rem}}
.cbtn{{padding:.2rem .6rem;border-radius:4px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.72rem}}
.cbtn.active{{border-color:#388bfd;color:#58a6ff}}
.cbtn:hover:not(.active){{background:#30363d;color:#e6edf3}}
#viewer{{flex:1;min-height:0;border-radius:4px;overflow:hidden;position:relative}}
.chart-wrap{{flex:1;min-height:0}}
.legend{{display:flex;gap:.9rem;margin-top:.3rem;flex-shrink:0;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:.3rem;font-size:.68rem;color:#8b949e}}
.ld{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
</style>
</head>
<body>
<header>
  <h1>AlphaFold Prediction &mdash; {job_name}</h1>
  <p>Chain A: {chain_a_len} aa &nbsp;|&nbsp; Chain B: {chain_b_len} aa &nbsp;|&nbsp; 5 models</p>
</header>
<div class="tabs">
  <span class="tabs-label">Model (ranked):</span>
  <div id="tab-container"></div>
</div>
<div class="stats" id="stats"></div>
<div class="main">
  <div id="viewer-wrap">
    <div class="panel-title">3D Structure</div>
    <div class="color-btns">
      <button class="cbtn active" onclick="setColor('plddt',this)">pLDDT</button>
      <button class="cbtn" onclick="setColor('chain',this)">Chain</button>
      <button class="cbtn" onclick="setColor('ss',this)">Sec. Structure</button>
      <button class="cbtn" onclick="setColor('rainbow',this)">Rainbow</button>
    </div>
    <div id="viewer"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Predicted Aligned Error (PAE)</div>
    <div class="chart-wrap" id="pae-chart"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Per-Residue Confidence (pLDDT)</div>
    <div class="chart-wrap" id="plddt-chart"></div>
    <div class="legend">
      <div class="li"><div class="ld" style="background:#0053D6"></div>Very high (&ge;90)</div>
      <div class="li"><div class="ld" style="background:#65CBF3"></div>High (70-90)</div>
      <div class="li"><div class="ld" style="background:#FFDB13"></div>Medium (50-70)</div>
      <div class="li"><div class="ld" style="background:#FF7D45"></div>Low (&lt;50)</div>
    </div>
  </div>
</div>

<script>
const MODELS = {data_js};
const CHAIN_A_LEN = {chain_a_len};
const CHAIN_B_LEN = {chain_b_len};
const SEQ_A = {json.dumps(seq_a)};
const SEQ_B = {json.dumps(seq_b)};

let currentIdx = 0;
let currentColor = 'plddt';
let viewer = null;

function plddtColor(b) {{
  if (b >= 90) return '#0053D6';
  if (b >= 70) return '#65CBF3';
  if (b >= 50) return '#FFDB13';
  return '#FF7D45';
}}

function buildTabs() {{
  const container = document.getElementById('tab-container');
  MODELS.forEach((m, rank) => {{
    const btn = document.createElement('button');
    btn.className = 'tab' + (rank === 0 ? ' active' : '');
    btn.innerHTML = `Rank ${{rank+1}} <span style="opacity:.7">(score: ${{m.ranking_score.toFixed(2)}})</span>`;
    btn.onclick = () => selectModel(rank, btn);
    container.appendChild(btn);
  }});
}}

function selectModel(rank, btn) {{
  currentIdx = rank;
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  updateStats();
  updateViewer();
  updatePAE();
  updatePLDDT();
}}

function updateStats() {{
  const m = MODELS[currentIdx];
  const el = document.getElementById('stats');
  const cp = m.chain_pair_iptm;
  el.innerHTML = `
    <div class="stat"><div class="stat-label">Ranking Score</div><div class="stat-value">${{m.ranking_score}}</div></div>
    <div class="stat"><div class="stat-label">ipTM</div><div class="stat-value">${{m.iptm}}</div></div>
    <div class="stat"><div class="stat-label">pTM</div><div class="stat-value">${{m.ptm}}</div></div>
    ${{cp && cp.length >= 2 ? `<div class="stat"><div class="stat-label">ipTM A&rarr;B</div><div class="stat-value">${{cp[0][1]}}</div></div>
    <div class="stat"><div class="stat-label">ipTM B&rarr;A</div><div class="stat-value">${{cp[1][0]}}</div></div>` : ''}}
    <div class="stat"><div class="stat-label">Model index</div><div class="stat-value">${{m.index}}</div></div>
  `;
}}

function initViewer() {{
  const el = document.getElementById('viewer');
  viewer = $3Dmol.createViewer(el, {{
    backgroundColor: '#0d1117',
    antialias: true,
  }});
}}

function applyColoring() {{
  if (currentColor === 'plddt') {{
    viewer.setStyle({{}}, {{cartoon: {{colorfunc: a => plddtColor(a.b)}}}});
  }} else if (currentColor === 'chain') {{
    viewer.setStyle({{chain: 'A'}}, {{cartoon: {{color: '#4C8CF5'}}}});
    viewer.setStyle({{chain: 'B'}}, {{cartoon: {{color: '#F07B4C'}}}});
  }} else if (currentColor === 'ss') {{
    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'ssJmol'}}}});
  }} else if (currentColor === 'rainbow') {{
    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'rainbow'}}}});
  }}
  viewer.render();
}}

function updateViewer() {{
  const m = MODELS[currentIdx];
  viewer.clear();
  viewer.addModel(m.cif, 'cif');
  applyColoring();
  viewer.zoomTo();
  viewer.render();
}}

function setColor(mode, btn) {{
  currentColor = mode;
  document.querySelectorAll('.cbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyColoring();
}}

function updatePAE() {{
  const m = MODELS[currentIdx];
  const n = m.pae.length;
  const boundary = CHAIN_A_LEN - 0.5;

  // Build axis labels
  const labelsA = Array.from({{length: CHAIN_A_LEN}}, (_, i) => 'A' + (i+1));
  const labelsB = Array.from({{length: CHAIN_B_LEN}}, (_, i) => 'B' + (i+1));
  const labels = [...labelsA, ...labelsB];

  // Tick positions every 20 residues
  const tickvals = [], ticktext = [];
  for (let i = 0; i < n; i += 20) {{
    tickvals.push(i);
    ticktext.push(labels[i]);
  }}

  const trace = {{
    z: m.pae,
    type: 'heatmap',
    colorscale: [[0,'#1a7a4a'],[0.15,'#4fb87e'],[0.4,'#b5e8c8'],[0.7,'#f5f5e8'],[1,'#f5f5e8']],
    zmin: 0, zmax: 30,
    colorbar: {{
      title: 'Å', titleside: 'right',
      thickness: 12, len: 0.8,
      tickfont: {{color:'#8b949e', size:10}},
      titlefont: {{color:'#8b949e', size:10}},
    }},
    hovertemplate: 'Scored: %{{y}}<br>Aligned: %{{x}}<br>PAE: %{{z:.1f}} Å<extra></extra>',
    xaxis: 'x', yaxis: 'y',
  }};

  const layout = {{
    paper_bgcolor: '#0d1117', plot_bgcolor: '#161b22',
    margin: {{l:45,r:60,t:10,b:45}},
    xaxis: {{
      title: {{text:'Aligned residue', font:{{color:'#8b949e',size:10}}}},
      tickvals, ticktext, tickfont:{{color:'#8b949e',size:9}},
      gridcolor:'#30363d', linecolor:'#30363d',
    }},
    yaxis: {{
      title: {{text:'Scored residue', font:{{color:'#8b949e',size:10}}}},
      tickvals, ticktext, tickfont:{{color:'#8b949e',size:9}},
      gridcolor:'#30363d', linecolor:'#30363d',
      autorange:'reversed',
    }},
    shapes: [
      {{type:'line',x0:boundary,x1:boundary,y0:-0.5,y1:n-0.5,line:{{color:'#e6edf3',width:1.5,dash:'dot'}}}},
      {{type:'line',y0:boundary,y1:boundary,x0:-0.5,x1:n-0.5,line:{{color:'#e6edf3',width:1.5,dash:'dot'}}}},
    ],
    annotations: [
      {{x:CHAIN_A_LEN/2,y:-2.5,text:'Chain A',showarrow:false,font:{{color:'#8b949e',size:9}},xref:'x',yref:'y'}},
      {{x:CHAIN_A_LEN+CHAIN_B_LEN/2,y:-2.5,text:'Chain B',showarrow:false,font:{{color:'#8b949e',size:9}},xref:'x',yref:'y'}},
    ],
  }};

  Plotly.react('pae-chart', [trace], layout, {{responsive:true, displayModeBar:false}});
}}

function updatePLDDT() {{
  const m = MODELS[currentIdx];
  const pa = m.plddt_a;
  const pb = m.plddt_b;

  function segmentedTrace(values, offset, chain, seqStr) {{
    // Split into colored segments by pLDDT band
    const x = values.map((_, i) => i + offset + 1);
    const bands = [
      {{min:90, max:101, color:'#0053D6', name:'Very high'}},
      {{min:70, max:90,  color:'#65CBF3', name:'High'}},
      {{min:50, max:70,  color:'#FFDB13', name:'Medium'}},
      {{min:0,  max:50,  color:'#FF7D45', name:'Low'}},
    ];
    return bands.map(band => {{
      const xb = [], yb = [], texts = [];
      values.forEach((v, i) => {{
        if (v >= band.min && v < band.max) {{
          xb.push(x[i]); yb.push(v);
          texts.push(`Chain ${{chain}} Res ${{i+1}}${{seqStr ? ' ('+seqStr[i]+')' : ''}}: ${{v.toFixed(1)}}`);
        }}
      }});
      return {{x:xb,y:yb,text:texts,type:'scatter',mode:'markers',marker:{{color:band.color,size:4,opacity:0.85}},
        name:`${{band.name}} (${{chain}})`,showlegend:false,hovertemplate:'%{{text}}<extra></extra>'}};
    }});
  }}

  // Also add line traces
  const lineA = {{x:pa.map((_,i)=>i+1),y:pa,type:'scatter',mode:'lines',line:{{color:'rgba(255,255,255,0.2)',width:1}},showlegend:false,hoverinfo:'skip'}};
  const lineB = {{x:pb.map((_,i)=>i+CHAIN_A_LEN+1),y:pb,type:'scatter',mode:'lines',line:{{color:'rgba(255,255,255,0.2)',width:1}},showlegend:false,hoverinfo:'skip'}};

  const traces = [lineA, lineB, ...segmentedTrace(pa, 0, 'A', SEQ_A), ...segmentedTrace(pb, CHAIN_A_LEN, 'B', SEQ_B)];

  const layout = {{
    paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
    margin:{{l:45,r:15,t:10,b:45}},
    xaxis:{{
      title:{{text:'Residue', font:{{color:'#8b949e',size:10}}}},
      tickfont:{{color:'#8b949e',size:9}},
      gridcolor:'#30363d', linecolor:'#30363d',
      range:[0.5, CHAIN_A_LEN + CHAIN_B_LEN + 0.5],
    }},
    yaxis:{{
      title:{{text:'pLDDT', font:{{color:'#8b949e',size:10}}}},
      tickfont:{{color:'#8b949e',size:9}},
      gridcolor:'#30363d', linecolor:'#30363d',
      range:[0,100],
    }},
    shapes:[
      {{type:'rect',x0:0.5,x1:CHAIN_A_LEN+0.5,y0:0,y1:100,fillcolor:'rgba(76,140,245,0.05)',line:{{width:0}}}},
      {{type:'rect',x0:CHAIN_A_LEN+0.5,x1:CHAIN_A_LEN+CHAIN_B_LEN+0.5,y0:0,y1:100,fillcolor:'rgba(240,123,76,0.05)',line:{{width:0}}}},
      {{type:'line',x0:CHAIN_A_LEN+0.5,x1:CHAIN_A_LEN+0.5,y0:0,y1:100,line:{{color:'#e6edf3',width:1,dash:'dot'}}}},
      {{type:'line',x0:0.5,x1:CHAIN_A_LEN+CHAIN_B_LEN+0.5,y0:90,y1:90,line:{{color:'#0053D6',width:1,dash:'dot'}}}},
      {{type:'line',x0:0.5,x1:CHAIN_A_LEN+CHAIN_B_LEN+0.5,y0:70,y1:70,line:{{color:'#65CBF3',width:1,dash:'dot'}}}},
      {{type:'line',x0:0.5,x1:CHAIN_A_LEN+CHAIN_B_LEN+0.5,y0:50,y1:50,line:{{color:'#FFDB13',width:1,dash:'dot'}}}},
    ],
    annotations:[
      {{x:CHAIN_A_LEN/2,y:5,text:'Chain A',showarrow:false,font:{{color:'#4C8CF5',size:9}},xref:'x',yref:'y'}},
      {{x:CHAIN_A_LEN+CHAIN_B_LEN/2,y:5,text:'Chain B',showarrow:false,font:{{color:'#F07B4C',size:9}},xref:'x',yref:'y'}},
    ],
  }};

  Plotly.react('plddt-chart', traces, layout, {{responsive:true, displayModeBar:false}});
}}

window.addEventListener('DOMContentLoaded', () => {{
  buildTabs();
  initViewer();
  updateStats();
  updateViewer();
  updatePAE();
  updatePLDDT();

  window.addEventListener('resize', () => {{
    Plotly.Plots.resize('pae-chart');
    Plotly.Plots.resize('plddt-chart');
    if (viewer) viewer.resize();
  }});
}});
</script>
</body>
</html>
"""


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else "fold_run_4_long.zip"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "visualization.html"

    print(f"Loading {zip_path}...")
    models, job_name, chain_a_len, chain_b_len, seq_a, seq_b = load_models(zip_path)
    print(f"  Job: {job_name} | {len(models)} models | Chain A: {chain_a_len} aa, Chain B: {chain_b_len} aa")

    print("Generating HTML...")
    html = generate_html(models, job_name, chain_a_len, chain_b_len, seq_a, seq_b)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Done! Open in your browser: {out_path}")


if __name__ == "__main__":
    main()
