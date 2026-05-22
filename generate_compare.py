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

import numpy as np


# ── CIF helpers ───────────────────────────────────────────────────────────────

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


def parse_ca_coords(cif_text: str, chain: str = "A") -> dict:
    """Return {resnum: np.array([x,y,z])} for Cα atoms of the given chain."""
    coords = {}
    for line in cif_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            if parts[3] == "CA" and parts[6] == chain:
                resnum = int(parts[8])
                coords[resnum] = np.array([float(parts[10]), float(parts[11]), float(parts[12])])
        except (ValueError, IndexError):
            continue
    return coords


def kabsch_superimpose(ref_cif: str, mob_cif: str) -> str:
    """
    Superimpose mob_cif onto ref_cif using shared chain A Cα atoms (Kabsch algorithm).
    Returns mob_cif text with transformed coordinates.
    """
    ref_ca = parse_ca_coords(ref_cif, "A")
    mob_ca = parse_ca_coords(mob_cif, "A")

    common = sorted(set(ref_ca) & set(mob_ca))
    P = np.array([ref_ca[r] for r in common])   # reference (fixed)
    Q = np.array([mob_ca[r] for r in common])   # mobile (to be rotated)

    c_P = P.mean(axis=0)
    c_Q = Q.mean(axis=0)
    H = (Q - c_Q).T @ (P - c_P)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # rotation matrix
    t = c_P - c_Q @ R.T                         # translation

    rmsd = float(np.sqrt(((( Q @ R.T + t) - P) ** 2).sum(axis=1).mean()))
    print(f"  Chain A RMSD after superimposition: {rmsd:.2f} Å ({len(common)} Cα atoms)")

    # Apply R, t to every atom in mob_cif
    out_lines = []
    for line in mob_cif.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            parts = line.split()
            if len(parts) >= 13:
                try:
                    xyz = np.array([float(parts[10]), float(parts[11]), float(parts[12])])
                    xyz2 = xyz @ R.T + t
                    parts[10] = f"{xyz2[0]:.3f}"
                    parts[11] = f"{xyz2[1]:.3f}"
                    parts[12] = f"{xyz2[2]:.3f}"
                    out_lines.append(" ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass
        out_lines.append(line)
    return "\n".join(out_lines)


# ── data loading ──────────────────────────────────────────────────────────────

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

        best_i, best_score = indices[0], -1.0
        for i in indices:
            sc = json.loads(zf.read(f"{prefix}_summary_confidences_{i}.json"))
            if sc["ranking_score"] > best_score:
                best_score, best_i = sc["ranking_score"], i

        cif     = zf.read(f"{prefix}_model_{best_i}.cif").decode()
        summary = json.loads(zf.read(f"{prefix}_summary_confidences_{best_i}.json"))
        full    = json.loads(zf.read(f"{prefix}_full_data_{best_i}.json"))
        plddt_a, plddt_b = parse_cif_residue_plddts(cif)

        return {
            "name":          req.get("name", prefix),
            "cif":           cif,
            "ranking_score": round(summary["ranking_score"], 3),
            "iptm":          round(summary.get("iptm", 0), 3),
            "ptm":           round(summary.get("ptm", 0), 3),
            "plddt_a":       plddt_a,
            "plddt_b":       plddt_b,
            "seq_a":         seqs[0] if seqs else "",
            "seq_b":         seqs[1] if len(seqs) > 1 else "",
            "chain_a_len":   sum(1 for c in full["token_chain_ids"] if c == "A"),
            "chain_b_len":   sum(1 for c in full["token_chain_ids"] if c == "B"),
        }


# ── sequence diff ─────────────────────────────────────────────────────────────

def compute_diff(seq_short: str, seq_long: str) -> dict:
    """Align seq_short (ref) against seq_long (alt). Returns per-residue annotations."""
    m = difflib.SequenceMatcher(None, seq_long, seq_short, autojunk=False)

    short_annot = ["match"] * (len(seq_short) + 1)
    long_annot  = ["match"] * (len(seq_long)  + 1)

    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op == "delete":                    # in long only
            for i in range(i1, i2):
                long_annot[i + 1] = "ins"
        elif op == "replace":
            for i in range(i1, i2):
                long_annot[i + 1] = "sub"
            for j in range(j1, j2):
                short_annot[j + 1] = "sub"

    # Build alignment display strings
    la, sa, mk = [], [], []
    pi = pj = 0
    for bi, bj, bsize in m.get_matching_blocks():
        gi, gj = seq_long[pi:bi], seq_short[pj:bj]
        mg = max(len(gi), len(gj))
        if mg:
            la.append(gi.ljust(mg, "-"))
            sa.append(gj.ljust(mg, "-"))
            mk.append(" " * mg)
        if bsize:
            la.append(seq_long[bi:bi+bsize])
            sa.append(seq_short[bj:bj+bsize])
            mk.append("|" * bsize)
        pi, pj = bi + bsize, bj + bsize

    return {
        "short_annot": short_annot,
        "long_annot":  long_annot,
        "aln_long":    "".join(la),
        "aln_short":   "".join(sa),
        "aln_mark":    "".join(mk),
    }


def build_seq_html(diff: dict) -> str:
    aln_long  = diff["aln_long"]
    aln_short = diff["aln_short"]

    def span(char, cls):
        return f'<span class="gap">-</span>' if char == "-" else f'<span class="{cls}">{char}</span>'

    long_html, short_html = [], []
    li = lj = 0

    for pos in range(len(aln_long)):
        lc, sc = aln_long[pos], aln_short[pos]
        li += lc != "-"
        lj += sc != "-"
        long_html.append(span(lc,  diff["long_annot"][li]  if lc != "-" else "gap"))
        short_html.append(span(sc, diff["short_annot"][lj] if sc != "-" else "gap"))

    return (
        '<div class="seq-row"><span class="seq-label">Long&nbsp;&nbsp;</span>'  + "".join(long_html)  + "</div>"
        '<div class="seq-row"><span class="seq-label">Short&nbsp;</span>' + "".join(short_html) + "</div>"
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

def generate_html(short: dict, long_: dict, diff: dict) -> str:
    seq_panel = build_seq_html(diff)

    long_ins_res  = [i for i, a in enumerate(diff["long_annot"])  if i > 0 and a == "ins"]
    long_sub_res  = [i for i, a in enumerate(diff["long_annot"])  if i > 0 and a == "sub"]
    short_sub_res = [i for i, a in enumerate(diff["short_annot"]) if i > 0 and a == "sub"]

    short_js = json.dumps({k: short[k] for k in ["cif","plddt_a","plddt_b","chain_a_len","chain_b_len","seq_a","seq_b"]})
    long_js  = json.dumps({k: long_[k] for k in ["cif","plddt_a","plddt_b","chain_a_len","chain_b_len","seq_a","seq_b"]})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Allele Comparison — {short["name"]} vs {long_["name"]}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{padding:.6rem 1.2rem;border-bottom:1px solid #30363d;flex-shrink:0;display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap}}
header h1{{font-size:1rem;font-weight:600}}
header p{{font-size:.78rem;color:#8b949e}}
.controls{{display:flex;gap:.5rem;padding:.4rem 1.2rem;border-bottom:1px solid #30363d;background:#161b22;flex-shrink:0;align-items:center;flex-wrap:wrap}}
.clabel{{font-size:.72rem;color:#8b949e;margin-right:.1rem}}
.cbtn{{padding:.22rem .6rem;border-radius:4px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.72rem}}
.cbtn.active{{border-color:#388bfd;color:#58a6ff;background:#0d1f3a}}
.cbtn:hover:not(.active){{background:#30363d;color:#e6edf3}}
.sep{{width:1px;background:#30363d;align-self:stretch;margin:0 .3rem}}
.sync-btn{{padding:.22rem .6rem;border-radius:4px;border:1px solid #30363d;background:#21262d;color:#8b949e;cursor:pointer;font-size:.72rem}}
.sync-btn.on{{border-color:#3fb950;color:#3fb950;background:#0d2f17}}
.legend{{display:flex;gap:.75rem;align-items:center;margin-left:auto;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:.28rem;font-size:.67rem;color:#8b949e}}
.ld{{width:8px;height:8px;border-radius:2px;flex-shrink:0}}
.seq-panel{{padding:.35rem 1.2rem;border-bottom:1px solid #30363d;background:#0d1117;flex-shrink:0;overflow-x:auto}}
.seq-title{{font-size:.63rem;color:#8b949e;margin-bottom:.15rem;font-style:italic}}
.seq-row{{font-family:'Courier New',monospace;font-size:.7rem;line-height:1.55;white-space:nowrap;display:flex;align-items:center}}
.seq-label{{color:#556070;min-width:50px;font-size:.63rem;flex-shrink:0}}
span.ins{{color:#FF6B35;font-weight:600}}
span.sub{{color:#FFD700;font-weight:600}}
span.match{{color:#4a9060}}
span.gap{{color:#30363d}}
.stats-grid{{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #30363d;flex-shrink:0}}
.stats-col{{display:flex;gap:1.2rem;padding:.32rem 1.2rem;background:#161b22;flex-wrap:wrap;align-items:center}}
.stats-col:first-child{{border-right:1px solid #30363d}}
.stat{{display:flex;flex-direction:column}}
.stat-label{{font-size:.6rem;text-transform:uppercase;letter-spacing:.05em;color:#8b949e}}
.stat-value{{font-size:.82rem;font-weight:600}}
.viewers{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#30363d;flex:1;min-height:0}}
.viewer-wrap{{background:#0d1117;display:flex;flex-direction:column;min-height:0;padding:.45rem .5rem .5rem}}
.viewer-title{{font-size:.7rem;color:#8b949e;margin-bottom:.25rem;flex-shrink:0;font-weight:500;padding-left:.2rem}}
.viewer-div{{flex:1;min-height:0;border-radius:3px;overflow:hidden}}
</style>
</head>
<body>

<header>
  <h1>Allele Comparison</h1>
  <p>{short["name"]} vs {long_["name"]} &ensp;&mdash;&ensp; Chain A identical (121 aa) &nbsp;|&nbsp; Chain B: {long_["chain_b_len"]} aa (long) vs {short["chain_b_len"]} aa (short) &nbsp;|&nbsp; Structures superimposed on chain A</p>
</header>

<div class="controls">
  <span class="clabel">Color:</span>
  <button class="cbtn active" onclick="setColor('diff',this)">Differences</button>
  <button class="cbtn" onclick="setColor('plddt',this)">pLDDT</button>
  <button class="cbtn" onclick="setColor('chain',this)">Chain</button>
  <div class="sep"></div>
  <button class="sync-btn on" id="sync-btn" onclick="toggleSync()">Sync rotation: ON</button>
  <div class="legend">
    <div class="li"><div class="ld" style="background:#6c8ebf"></div>Chain A</div>
    <div class="li"><div class="ld" style="background:#4a9060"></div>Chain B (identical)</div>
    <div class="li"><div class="ld" style="background:#FF6B35"></div>Extra residues (long only)</div>
    <div class="li"><div class="ld" style="background:#FFD700"></div>Substitution</div>
  </div>
</div>

<div class="seq-panel">
  <div class="seq-title">Chain B sequence alignment</div>
  {seq_panel}
</div>

<div class="stats-grid">
  <div class="stats-col">
    <div class="stat"><div class="stat-label">Allele</div><div class="stat-value" style="color:#6c9ed4">{short["name"]}</div></div>
    <div class="stat"><div class="stat-label">Ranking Score</div><div class="stat-value">{short["ranking_score"]}</div></div>
    <div class="stat"><div class="stat-label">ipTM</div><div class="stat-value">{short["iptm"]}</div></div>
    <div class="stat"><div class="stat-label">pTM</div><div class="stat-value">{short["ptm"]}</div></div>
  </div>
  <div class="stats-col">
    <div class="stat"><div class="stat-label">Allele</div><div class="stat-value" style="color:#d47a6c">{long_["name"]}</div></div>
    <div class="stat"><div class="stat-label">Ranking Score</div><div class="stat-value">{long_["ranking_score"]}</div></div>
    <div class="stat"><div class="stat-label">ipTM</div><div class="stat-value">{long_["iptm"]}</div></div>
    <div class="stat"><div class="stat-label">pTM</div><div class="stat-value">{long_["ptm"]}</div></div>
  </div>
</div>

<div class="viewers">
  <div class="viewer-wrap">
    <div class="viewer-title" style="color:#6c9ed4">{short["name"]} — Chain B: {short["chain_b_len"]} aa</div>
    <div class="viewer-div" id="viewer-short"></div>
  </div>
  <div class="viewer-wrap">
    <div class="viewer-title" style="color:#d47a6c">{long_["name"]} — Chain B: {long_["chain_b_len"]} aa</div>
    <div class="viewer-div" id="viewer-long"></div>
  </div>
</div>

<script>
const SHORT = {short_js};
const LONG  = {long_js};
const LONG_INS  = {json.dumps(long_ins_res)};
const LONG_SUB  = {json.dumps(long_sub_res)};
const SHORT_SUB = {json.dumps(short_sub_res)};

let vShort = null, vLong = null;
let colorMode = 'diff';
let syncOn = true;
let syncing = false;

function plddtColor(b) {{
  if (b >= 90) return '#0053D6';
  if (b >= 70) return '#65CBF3';
  if (b >= 50) return '#FFDB13';
  return '#FF7D45';
}}

function applyStyle(v, subRes, insRes) {{
  if (colorMode === 'plddt') {{
    v.setStyle({{}}, {{cartoon: {{colorfunc: a => plddtColor(a.b)}}}});
  }} else if (colorMode === 'chain') {{
    v.setStyle({{chain: 'A'}}, {{cartoon: {{color: '#6c8ebf'}}}});
    v.setStyle({{chain: 'B'}}, {{cartoon: {{color: '#9b6cbf'}}}});
  }} else {{
    // diff mode: base colors
    v.setStyle({{chain: 'A'}}, {{cartoon: {{color: '#6c8ebf'}}}});
    v.setStyle({{chain: 'B'}}, {{cartoon: {{color: '#4a9060'}}}});
  }}
  // Difference highlights always shown in diff mode
  if (colorMode === 'diff') {{
    if (insRes.length) {{
      v.setStyle({{chain: 'B', resi: insRes.join(',')}}, {{cartoon: {{color: '#FF6B35'}}}});
    }}
    if (subRes.length) {{
      v.setStyle({{chain: 'B', resi: subRes.join(',')}}, {{cartoon: {{color: '#FFD700'}}}});
    }}
  }}
  v.render();
}}

function initViewers() {{
  const bg = '#0d1117';
  vShort = $3Dmol.createViewer(document.getElementById('viewer-short'), {{backgroundColor: bg, antialias: true}});
  vLong  = $3Dmol.createViewer(document.getElementById('viewer-long'),  {{backgroundColor: bg, antialias: true}});

  vShort.addModel(SHORT.cif, 'cif');
  applyStyle(vShort, SHORT_SUB, []);
  vShort.zoomTo();
  vShort.render();

  vLong.addModel(LONG.cif, 'cif');
  applyStyle(vLong, LONG_SUB, LONG_INS);
  vLong.zoomTo();
  vLong.render();

  // Start both viewers in the same orientation as the short structure
  setTimeout(() => {{
    const view = vShort.getView();
    vLong.setView(view);
    vLong.render();
  }}, 100);
}}

function setColor(mode, btn) {{
  colorMode = mode;
  document.querySelectorAll('.cbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyStyle(vShort, SHORT_SUB, []);
  applyStyle(vLong,  LONG_SUB,  LONG_INS);
}}

function toggleSync() {{
  syncOn = !syncOn;
  const btn = document.getElementById('sync-btn');
  btn.textContent = 'Sync rotation: ' + (syncOn ? 'ON' : 'OFF');
  btn.classList.toggle('on', syncOn);
}}

// Poll-based sync: mirror short → long rotation
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
    vShort && vShort.resize();
    vLong  && vLong.resize();
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

    print("Superimposing long onto short via chain A Kabsch alignment...")
    long_["cif"] = kabsch_superimpose(short["cif"], long_["cif"])

    print("Computing sequence diff...")
    diff = compute_diff(short["seq_b"], long_["seq_b"])
    ins  = [i for i, a in enumerate(diff["long_annot"])  if i > 0 and a == "ins"]
    sub  = [i for i, a in enumerate(diff["long_annot"])  if i > 0 and a == "sub"]
    print(f"  Extra residues in long (chain B res): {ins}")
    print(f"  Substitutions in long  (chain B res): {sub}")

    print("Generating HTML...")
    html = generate_html(short, long_, diff)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Done! Open in your browser: {out_path}")


if __name__ == "__main__":
    main()
