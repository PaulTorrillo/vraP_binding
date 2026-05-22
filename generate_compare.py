#!/usr/bin/env python3
"""
Generate a side-by-side allele comparison HTML from two AlphaFold Server zips.
Usage: python generate_compare.py zip_a zip_b [output.html]
Defaults: fold_run_2_short.zip fold_run_4_long.zip -> comparison.html
"""

import difflib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_cif_residue_plddts(cif_text: str):
    residue_atoms: dict = defaultdict(list)
    for line in cif_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            chain  = parts[6]
            resnum = int(parts[8])
            bfac   = float(parts[14])
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


def load_best_model(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        cif_files = [n for n in names if "_model_" in n and n.endswith(".cif")]
        prefix = cif_files[0].rsplit("_model_", 1)[0]
        indices = sorted(int(n.split("_model_")[1].split(".")[0]) for n in cif_files)

        req = json.loads(zf.read(next(n for n in names if n.endswith("_job_request.json"))))
        if isinstance(req, list):
            req = req[0]
        seqs = [s["proteinChain"]["sequence"]
                for s in req.get("sequences", []) if "proteinChain" in s]

        # Pick highest-ranked model
        best_i, best_score = indices[0], -1
        for i in indices:
            sc = json.loads(zf.read(f"{prefix}_summary_confidences_{i}.json"))
            if sc["ranking_score"] > best_score:
                best_score, best_i = sc["ranking_score"], i

        cif     = zf.read(f"{prefix}_model_{best_i}.cif").decode()
        summary = json.loads(zf.read(f"{prefix}_summary_confidences_{best_i}.json"))
        full    = json.loads(zf.read(f"{prefix}_full_data_{best_i}.json"))
        plddt_a, plddt_b = parse_cif_residue_plddts(cif)

        chain_a_len = sum(1 for c in full["token_chain_ids"] if c == "A")
        chain_b_len = sum(1 for c in full["token_chain_ids"] if c == "B")

        return {
            "name":         req.get("name", prefix),
            "zip":          zip_path,
            "cif":          cif,
            "ranking_score": round(summary["ranking_score"], 3),
            "iptm":          round(summary.get("iptm", 0), 3),
            "ptm":           round(summary.get("ptm", 0), 3),
            "chain_pair_iptm": summary.get("chain_pair_iptm", []),
            "pae":           full["pae"],
            "plddt_a":       plddt_a,
            "plddt_b":       plddt_b,
            "seq_a":         seqs[0] if len(seqs) > 0 else "",
            "seq_b":         seqs[1] if len(seqs) > 1 else "",
            "chain_a_len":   chain_a_len,
            "chain_b_len":   chain_b_len,
        }


def compute_diff(seq_ref: str, seq_alt: str) -> dict:
    """
    Align seq_ref (short) against seq_alt (long) and return per-residue
    annotation arrays (1-indexed residue numbers) for each sequence.
    Returns:
        ref_annot:  list of ('match'|'sub'|'del') per ref residue (1-indexed)
        alt_annot:  list of ('match'|'sub'|'ins'|'del') per alt residue (1-indexed)
        aln_ref, aln_alt, aln_mark: alignment strings for display
    """
    m = difflib.SequenceMatcher(None, seq_alt, seq_ref, autojunk=False)

    ref_annot = ["match"] * (len(seq_ref) + 1)   # index 1..len
    alt_annot = ["match"] * (len(seq_alt) + 1)

    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op == "equal":
            pass
        elif op == "insert":
            # present in ref only (deletion from alt perspective)
            for j in range(j1, j2):
                ref_annot[j + 1] = "del"
        elif op == "delete":
            # present in alt only (insertion = extra residues)
            for i in range(i1, i2):
                alt_annot[i + 1] = "ins"
        elif op == "replace":
            for i in range(i1, i2):
                alt_annot[i + 1] = "sub"
            for j in range(j1, j2):
                ref_annot[j + 1] = "sub"

    # Build display alignment strings
    blocks = m.get_matching_blocks()
    la, sa, mk = [], [], []
    pi, pj = 0, 0
    for bi, bj, bsize in blocks:
        gap_i = seq_alt[pi:bi]
        gap_j = seq_ref[pj:bj]
        maxg  = max(len(gap_i), len(gap_j))
        if maxg:
            la.append(gap_i.ljust(maxg, "-"))
            sa.append(gap_j.ljust(maxg, "-"))
            mk.append(" " * maxg)
        if bsize:
            la.append(seq_alt[bi:bi+bsize])
            sa.append(seq_ref[bj:bj+bsize])
            mk.append("|" * bsize)
        pi, pj = bi + bsize, bj + bsize

    return {
        "ref_annot": ref_annot,
        "alt_annot": alt_annot,
        "aln_ref":   "".join(sa),
        "aln_alt":   "".join(la),
        "aln_mark":  "".join(mk),
    }


def build_seq_panel_html(diff: dict, short_seq_b: str, long_seq_b: str) -> str:
    """Build coloured HTML sequence alignment for chain B."""
    aln_long  = diff["aln_alt"]
    aln_short = diff["aln_ref"]
    mark      = diff["aln_mark"]

    def span(char, cls):
        if char == "-":
            return f'<span class="gap">-</span>'
        return f'<span class="{cls}">{char}</span>'

    long_html  = []
    short_html = []
    li = lj = 0  # residue counters (0-based, for annotation lookup)

    for pos in range(len(aln_long)):
        lc = aln_long[pos]
        sc = aln_short[pos]

        if lc != "-":
            li += 1
            ann = diff["alt_annot"][li]
        else:
            ann = "gap"

        if sc != "-":
            lj += 1
            s_ann = diff["ref_annot"][lj]
        else:
            s_ann = "gap"

        long_html.append(span(lc, ann))
        short_html.append(span(sc, s_ann))

    return (
        '<div class="seq-row"><span class="seq-label">Long&nbsp;&nbsp;</span>'
        + "".join(long_html) + "</div>"
        '<div class="seq-row"><span class="seq-label">Short&nbsp;</span>'
        + "".join(short_html) + "</div>"
    )


# ── HTML generation ──────────────────────────────────────────────────────────

def generate_html(short: dict, long_: dict, diff: dict) -> str:
    seq_panel = build_seq_panel_html(diff, short["seq_b"], long_["seq_b"])

    # Residue numbers that are different (for 3Dmol highlighting)
    # alt = long, ref = short
    long_ins_res  = [i for i, a in enumerate(diff["alt_annot"]) if i > 0 and a == "ins"]
    long_sub_res  = [i for i, a in enumerate(diff["alt_annot"]) if i > 0 and a == "sub"]
    short_sub_res = [i for i, a in enumerate(diff["ref_annot"]) if i > 0 and a == "sub"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Allele Comparison — {short["name"]} vs {long_["name"]}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{padding:.6rem 1.2rem;border-bottom:1px solid #30363d;flex-shrink:0;display:flex;align-items:baseline;gap:1rem}}
header h1{{font-size:1rem;font-weight:600}}
header p{{font-size:.78rem;color:#8b949e}}
.controls{{display:flex;gap:.5rem;padding:.45rem 1.2rem;border-bottom:1px solid #30363d;background:#161b22;flex-shrink:0;align-items:center;flex-wrap:wrap}}
.controls-label{{font-size:.72rem;color:#8b949e;margin-right:.2rem}}
.cbtn{{padding:.25rem .65rem;border-radius:4px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.72rem}}
.cbtn.active{{border-color:#388bfd;color:#58a6ff}}
.cbtn:hover:not(.active){{background:#30363d;color:#e6edf3}}
.sep{{width:1px;background:#30363d;margin:0 .25rem;align-self:stretch}}
.sync-btn{{padding:.25rem .65rem;border-radius:4px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.72rem}}
.sync-btn.on{{border-color:#3fb950;color:#3fb950;background:#0d2f17}}
.legend{{display:flex;gap:.8rem;align-items:center;margin-left:auto}}
.li{{display:flex;align-items:center;gap:.3rem;font-size:.68rem;color:#8b949e}}
.ld{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}
.seq-panel{{padding:.4rem 1.2rem;border-bottom:1px solid #30363d;background:#0d1117;flex-shrink:0;overflow-x:auto}}
.seq-row{{font-family:monospace;font-size:.7rem;line-height:1.6;white-space:nowrap;display:flex;align-items:center}}
.seq-label{{color:#8b949e;min-width:52px;font-size:.65rem}}
span.ins{{color:#FF6B35;font-weight:700}}
span.sub{{color:#FFD700;font-weight:700}}
span.match{{color:#3fb950}}
span.del{{color:#8b949e;text-decoration:underline dotted}}
span.gap{{color:#30363d}}
.stats-grid{{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #30363d;flex-shrink:0}}
.stats-col{{display:flex;gap:1.2rem;padding:.35rem 1.2rem;background:#161b22;flex-wrap:wrap}}
.stats-col:first-child{{border-right:1px solid #30363d}}
.stat{{display:flex;flex-direction:column}}
.stat-label{{font-size:.62rem;text-transform:uppercase;letter-spacing:.05em;color:#8b949e}}
.stat-value{{font-size:.85rem;font-weight:600}}
.viewers{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#30363d;flex:1;min-height:0}}
.viewer-wrap{{background:#0d1117;display:flex;flex-direction:column;min-height:0;padding:.5rem}}
.viewer-title{{font-size:.72rem;color:#8b949e;margin-bottom:.3rem;flex-shrink:0;font-weight:500}}
.viewer-div{{flex:1;min-height:0;border-radius:3px;overflow:hidden;position:relative}}
</style>
</head>
<body>
<header>
  <h1>Allele Comparison</h1>
  <p>{short["name"]} &nbsp;vs&nbsp; {long_["name"]}&ensp;&mdash;&ensp;Chain A identical &nbsp;|&nbsp; Chain B: {long_["chain_b_len"]} aa (long) vs {short["chain_b_len"]} aa (short)</p>
</header>

<div class="controls">
  <span class="controls-label">Coloring:</span>
  <button class="cbtn active" onclick="setColor('plddt',this)">pLDDT</button>
  <button class="cbtn" onclick="setColor('chain',this)">Chain</button>
  <button class="cbtn" onclick="setColor('diff',this)">Differences</button>
  <div class="sep"></div>
  <button class="sync-btn" id="sync-btn" onclick="toggleSync()">Sync rotation: OFF</button>
  <div class="legend">
    <div class="li"><div class="ld" style="background:#FF6B35"></div>Extra residues (long only)</div>
    <div class="li"><div class="ld" style="background:#FFD700"></div>Substitution</div>
    <div class="li"><div class="ld" style="background:#3fb950"></div>Identical</div>
  </div>
</div>

<div class="seq-panel">
  <div style="font-size:.65rem;color:#8b949e;margin-bottom:.2rem;font-style:italic">Chain B alignment</div>
  {seq_panel}
</div>

<div class="stats-grid">
  <div class="stats-col">
    <div class="stat"><div class="stat-label">Allele</div><div class="stat-value" style="color:#4C8CF5">{short["name"]}</div></div>
    <div class="stat"><div class="stat-label">Ranking Score</div><div class="stat-value">{short["ranking_score"]}</div></div>
    <div class="stat"><div class="stat-label">ipTM</div><div class="stat-value">{short["iptm"]}</div></div>
    <div class="stat"><div class="stat-label">pTM</div><div class="stat-value">{short["ptm"]}</div></div>
  </div>
  <div class="stats-col">
    <div class="stat"><div class="stat-label">Allele</div><div class="stat-value" style="color:#F07B4C">{long_["name"]}</div></div>
    <div class="stat"><div class="stat-label">Ranking Score</div><div class="stat-value">{long_["ranking_score"]}</div></div>
    <div class="stat"><div class="stat-label">ipTM</div><div class="stat-value">{long_["iptm"]}</div></div>
    <div class="stat"><div class="stat-label">pTM</div><div class="stat-value">{long_["ptm"]}</div></div>
  </div>
</div>

<div class="viewers">
  <div class="viewer-wrap">
    <div class="viewer-title" style="color:#4C8CF5">{short["name"]} &mdash; Chain B: {short["chain_b_len"]} aa</div>
    <div class="viewer-div" id="viewer-short"></div>
  </div>
  <div class="viewer-wrap">
    <div class="viewer-title" style="color:#F07B4C">{long_["name"]} &mdash; Chain B: {long_["chain_b_len"]} aa</div>
    <div class="viewer-div" id="viewer-long"></div>
  </div>
</div>

<script>
const SHORT = {json.dumps({k: short[k] for k in ["cif","plddt_a","plddt_b","chain_a_len","chain_b_len","seq_a","seq_b"]})};
const LONG  = {json.dumps({k: long_[k] for k in ["cif","plddt_a","plddt_b","chain_a_len","chain_b_len","seq_a","seq_b"]})};

// Residue numbers of differences (1-indexed, chain B)
const LONG_INS  = {json.dumps(long_ins_res)};   // extra residues in long only
const LONG_SUB  = {json.dumps(long_sub_res)};   // substitution in long
const SHORT_SUB = {json.dumps(short_sub_res)};  // substitution in short

let vShort = null, vLong = null;
let colorMode = 'plddt';
let syncOn = false;
let syncing = false;

function plddtColor(b) {{
  if (b >= 90) return '#0053D6';
  if (b >= 70) return '#65CBF3';
  if (b >= 50) return '#FFDB13';
  return '#FF7D45';
}}

function initViewers() {{
  const bg = '#0d1117';
  vShort = $3Dmol.createViewer(document.getElementById('viewer-short'), {{backgroundColor: bg, antialias: true}});
  vLong  = $3Dmol.createViewer(document.getElementById('viewer-long'),  {{backgroundColor: bg, antialias: true}});
  loadAndRender(vShort, SHORT, SHORT_SUB, []);
  loadAndRender(vLong,  LONG,  LONG_SUB,  LONG_INS);
}}

function loadAndRender(v, data, subRes, insRes) {{
  v.clear();
  v.addModel(data.cif, 'cif');
  applyStyle(v, data, subRes, insRes);
  v.zoomTo();
  v.render();
}}

function applyStyle(v, data, subRes, insRes) {{
  if (colorMode === 'plddt') {{
    v.setStyle({{}}, {{cartoon: {{colorfunc: a => plddtColor(a.b)}}}});
  }} else if (colorMode === 'chain') {{
    v.setStyle({{chain: 'A'}}, {{cartoon: {{color: '#4C8CF5'}}}});
    v.setStyle({{chain: 'B'}}, {{cartoon: {{color: '#9b59b6'}}}});
  }} else if (colorMode === 'diff') {{
    // base: chain A gray, chain B green
    v.setStyle({{chain: 'A'}}, {{cartoon: {{color: '#555e6a'}}}});
    v.setStyle({{chain: 'B'}}, {{cartoon: {{color: '#3fb950'}}}});
  }}

  // Always overlay difference highlighting on chain B
  if (insRes.length) {{
    const resi = insRes.join(',');
    v.setStyle({{chain: 'B', resi: resi}}, {{
      cartoon: {{color: '#FF6B35'}},
      stick:   {{color: '#FF6B35', radius: 0.15}},
    }});
    v.addStyle({{chain: 'B', resi: resi}}, {{sphere: {{color: '#FF6B35', radius: 0.6}}}});
  }}
  if (subRes.length) {{
    const resi = subRes.join(',');
    v.setStyle({{chain: 'B', resi: resi}}, {{
      cartoon: {{color: '#FFD700'}},
      stick:   {{color: '#FFD700', radius: 0.2}},
    }});
    v.addStyle({{chain: 'B', resi: resi}}, {{sphere: {{color: '#FFD700', radius: 0.8}}}});
  }}

  v.render();
}}

function setColor(mode, btn) {{
  colorMode = mode;
  document.querySelectorAll('.cbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyStyle(vShort, SHORT, SHORT_SUB, []);
  applyStyle(vLong,  LONG,  LONG_SUB,  LONG_INS);
}}

function toggleSync() {{
  syncOn = !syncOn;
  const btn = document.getElementById('sync-btn');
  btn.textContent = 'Sync rotation: ' + (syncOn ? 'ON' : 'OFF');
  btn.classList.toggle('on', syncOn);
}}

// Sync loop: if sync is on, mirror short → long
let lastView = null;
function syncLoop() {{
  if (syncOn && vShort && vLong && !syncing) {{
    const v = vShort.getView();
    const vs = JSON.stringify(v);
    if (vs !== lastView) {{
      lastView = vs;
      syncing = true;
      vLong.setView(v);
      vLong.render();
      syncing = false;
    }}
  }}
  requestAnimationFrame(syncLoop);
}}

window.addEventListener('DOMContentLoaded', () => {{
  initViewers();
  syncLoop();
  window.addEventListener('resize', () => {{
    if (vShort) vShort.resize();
    if (vLong)  vLong.resize();
  }});
}});
</script>
</body>
</html>
"""


def main():
    zip_short = sys.argv[1] if len(sys.argv) > 1 else "fold_run_2_short.zip"
    zip_long  = sys.argv[2] if len(sys.argv) > 2 else "fold_run_4_long.zip"
    out_path  = sys.argv[3] if len(sys.argv) > 3 else "comparison.html"

    print(f"Loading {zip_short}...")
    short = load_best_model(zip_short)
    print(f"  {short['name']}: chain B {short['chain_b_len']} aa, ranking {short['ranking_score']}")

    print(f"Loading {zip_long}...")
    long_ = load_best_model(zip_long)
    print(f"  {long_['name']}: chain B {long_['chain_b_len']} aa, ranking {long_['ranking_score']}")

    print("Computing sequence diff...")
    diff = compute_diff(short["seq_b"], long_["seq_b"])
    ins  = [i for i, a in enumerate(diff["alt_annot"]) if i > 0 and a == "ins"]
    sub  = [i for i, a in enumerate(diff["alt_annot"]) if i > 0 and a == "sub"]
    print(f"  Long-only residues (ins): {ins}")
    print(f"  Substitutions in long   : {sub}")
    print(f"  Alignment preview:")
    print(f"  Long : {diff['aln_alt'][:80]}")
    print(f"         {diff['aln_mark'][:80]}")
    print(f"  Short: {diff['aln_ref'][:80]}")

    print("Generating HTML...")
    html = generate_html(short, long_, diff)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Done! Open in your browser: {out_path}")


if __name__ == "__main__":
    main()
