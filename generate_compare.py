#!/usr/bin/env python3
"""
Side-by-side allele comparison viewer from two AlphaFold Server zips.
Usage: python generate_compare.py [short.zip] [long.zip] [out.html]
"""

import difflib, json, sys, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np


# ── CIF helpers ───────────────────────────────────────────────────────────────

def best_model_cif(zip_path):
    """Return (name, seq_a, seq_b, cif_text, summary) for the top-ranked model."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        cifs  = [n for n in names if "_model_" in n and n.endswith(".cif")]
        pfx   = cifs[0].rsplit("_model_", 1)[0]
        idxs  = sorted(int(n.split("_model_")[1].split(".")[0]) for n in cifs)

        req = json.loads(zf.read(next(n for n in names if n.endswith("_job_request.json"))))
        if isinstance(req, list): req = req[0]
        seqs = [s["proteinChain"]["sequence"] for s in req.get("sequences", []) if "proteinChain" in s]

        best_i, best_sc = idxs[0], -1.0
        for i in idxs:
            sc = json.loads(zf.read(f"{pfx}_summary_confidences_{i}.json"))["ranking_score"]
            if sc > best_sc: best_sc, best_i = sc, i

        cif     = zf.read(f"{pfx}_model_{best_i}.cif").decode()
        summary = json.loads(zf.read(f"{pfx}_summary_confidences_{best_i}.json"))
        return (req.get("name", pfx),
                seqs[0] if seqs else "",
                seqs[1] if len(seqs) > 1 else "",
                cif,
                summary)


def ca_coords(cif_text, chain="A"):
    out = {}
    for line in cif_text.splitlines():
        if not line.startswith("ATOM"): continue
        p = line.split()
        if len(p) < 13: continue
        try:
            if p[3] == "CA" and p[6] == chain:
                out[int(p[8])] = np.array([float(p[10]), float(p[11]), float(p[12])])
        except (ValueError, IndexError): pass
    return out


def kabsch(ref_cif, mob_cif):
    """Superimpose mob_cif onto ref_cif using chain-A Cα. Returns transformed text."""
    P = np.array(list(ca_coords(ref_cif, "A").values()))
    Qd = ca_coords(mob_cif, "A")
    # use residues common to both
    ref_d = ca_coords(ref_cif, "A")
    common = sorted(set(ref_d) & set(Qd))
    P = np.array([ref_d[r] for r in common])
    Q = np.array([Qd[r]   for r in common])

    cP, cQ = P.mean(0), Q.mean(0)
    H = (Q - cQ).T @ (P - cP)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    t = cP - cQ @ R.T
    rmsd = float(np.sqrt((((Q @ R.T + t) - P)**2).sum(1).mean()))
    print(f"  Chain-A superimposition RMSD: {rmsd:.2f} Å ({len(common)} Cα)")

    lines = []
    for line in mob_cif.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            p = line.split()
            if len(p) >= 13:
                try:
                    xyz = np.array([float(p[10]), float(p[11]), float(p[12])]) @ R.T + t
                    p[10], p[11], p[12] = f"{xyz[0]:.3f}", f"{xyz[1]:.3f}", f"{xyz[2]:.3f}"
                    lines.append(" ".join(p)); continue
                except (ValueError, IndexError): pass
        lines.append(line)
    return "\n".join(lines)


# ── sequence diff ─────────────────────────────────────────────────────────────

def seq_diff(short_b, long_b):
    """Return (short_annot, long_annot, aln_short, aln_long) for chain B."""
    m = difflib.SequenceMatcher(None, long_b, short_b, autojunk=False)
    sa = ["match"] * (len(short_b) + 1)
    la = ["match"] * (len(long_b)  + 1)
    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op == "delete":
            for i in range(i1, i2): la[i+1] = "ins"
        elif op == "replace":
            for i in range(i1, i2): la[i+1] = "sub"
            for j in range(j1, j2): sa[j+1] = "sub"

    al, as_, mk = [], [], []
    pi = pj = 0
    for bi, bj, bs in m.get_matching_blocks():
        gi, gj = long_b[pi:bi], short_b[pj:bj]
        mg = max(len(gi), len(gj))
        if mg:
            al.append(gi.ljust(mg, "-")); as_.append(gj.ljust(mg, "-")); mk.append(" "*mg)
        if bs:
            al.append(long_b[bi:bi+bs]); as_.append(short_b[bj:bj+bs]); mk.append("|"*bs)
        pi, pj = bi+bs, bj+bs

    return sa, la, "".join(as_), "".join(al)


def seq_html(sa, la, aln_s, aln_l):
    def sp(c, cls): return f'<span class="gap">&#8209;</span>' if c=="-" else f'<span class="{cls}">{c}</span>'
    lh, sh = [], []
    li = lj = 0
    for lc, sc in zip(aln_l, aln_s):
        if lc != "-": li += 1
        if sc != "-": lj += 1
        lh.append(sp(lc, la[li] if lc!="-" else "gap"))
        sh.append(sp(sc, sa[lj] if sc!="-" else "gap"))
    return (
        '<div class="sr"><span class="sl">Long&nbsp;</span>'  + "".join(lh) + "</div>"
        '<div class="sr"><span class="sl">Short</span>' + "".join(sh) + "</div>"
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

def make_html(short_name, long_name, short_b_len, long_b_len,
              short_cif, long_cif,
              short_sum, long_sum,
              long_ins, long_sub, short_sub,
              seq_panel):

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{short_name} vs {long_name}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; overflow: hidden; }}

/* fixed top band */
#top {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 20;
  background: #0d1117; border-bottom: 2px solid #30363d;
}}
#hdr {{ padding: .5rem 1rem; display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; border-bottom: 1px solid #30363d; }}
#hdr h1 {{ font-size: .9rem; font-weight: 600; white-space: nowrap; }}
#hdr p  {{ font-size: .72rem; color: #8b949e; }}
#bar {{ display: flex; align-items: center; gap: .6rem; padding: .3rem 1rem;
        background: #161b22; border-bottom: 1px solid #30363d; flex-wrap: wrap; }}
.sbtn {{ padding: .2rem .6rem; border-radius: 4px; border: 1px solid #3fb950;
         background: #0d2f17; color: #3fb950; cursor: pointer; font-size: .7rem; font-family: inherit; }}
.sbtn.off {{ border-color: #30363d; background: #21262d; color: #8b949e; }}
.leg {{ display: flex; gap: .6rem; margin-left: auto; flex-wrap: wrap; }}
.li  {{ display: flex; align-items: center; gap: .25rem; font-size: .65rem; color: #8b949e; }}
.ld  {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
#seq {{ padding: .25rem 1rem; overflow-x: auto; border-bottom: 1px solid #30363d; }}
.st  {{ font-size: .6rem; color: #556070; margin-bottom: .08rem; }}
.sr  {{ font-family: 'Courier New', monospace; font-size: .67rem; line-height: 1.5;
        white-space: nowrap; display: flex; align-items: center; }}
.sl  {{ color: #444f60; min-width: 44px; font-size: .6rem; flex-shrink: 0; }}
span.ins   {{ color: #FF6B35; font-weight: 700; }}
span.sub   {{ color: #FFD700; font-weight: 700; }}
span.match {{ color: #4a6a8a; }}
span.gap   {{ color: #1e2530; }}
#stats {{ display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid #30363d; }}
.sc  {{ display: flex; gap: 1rem; padding: .28rem 1rem; background: #161b22; align-items: center; }}
.sc:first-child {{ border-right: 1px solid #30363d; }}
.sv  {{ display: flex; flex-direction: column; }}
.sk  {{ font-size: .57rem; text-transform: uppercase; letter-spacing: .05em; color: #8b949e; }}
.sv2 {{ font-size: .78rem; font-weight: 600; }}

/* viewers: fixed below top band, two columns */
#viewers {{
  position: fixed; left: 0; right: 0; bottom: 0;
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 2px; background: #21262d;
}}
.vw {{ background: #0d1117; position: relative; overflow: hidden; }}
.vtitle {{
  position: absolute; top: 0; left: 0; right: 0; height: 26px;
  padding: .25rem .6rem; font-size: .68rem; font-weight: 500;
  background: rgba(13,17,23,.92); z-index: 5;
  border-bottom: 1px solid #21262d;
}}
.vbox {{
  position: absolute; left: 0; right: 0; bottom: 0; top: 26px;
}}
</style>
</head>
<body>

<div id="top">
  <div id="hdr">
    <h1>Allele Comparison</h1>
    <p>Chain A identical (121 aa) &nbsp;·&nbsp; Chain B: {long_b_len} aa (long) vs {short_b_len} aa (short) &nbsp;·&nbsp; Aligned on chain A</p>
  </div>
  <div id="bar">
    <button class="sbtn" id="sbtn" onclick="toggleSync()">Sync: ON</button>
    <div class="leg">
      <div class="li"><div class="ld" style="background:#5b8dd9"></div>Chain A</div>
      <div class="li"><div class="ld" style="background:#9b6cbf"></div>Chain B</div>
      <div class="li"><div class="ld" style="background:#FF6B35"></div>Extra residues (long)</div>
      <div class="li"><div class="ld" style="background:#FFD700"></div>Substitution</div>
    </div>
  </div>
  <div id="seq">
    <div class="st">Chain B</div>
    {seq_panel}
  </div>
  <div id="stats">
    <div class="sc">
      <div class="sv"><div class="sk">Allele</div><div class="sv2" style="color:#7aade8">{short_name}</div></div>
      <div class="sv"><div class="sk">Ranking</div><div class="sv2">{short_sum['ranking_score']:.3f}</div></div>
      <div class="sv"><div class="sk">ipTM</div><div class="sv2">{short_sum.get('iptm',0):.3f}</div></div>
      <div class="sv"><div class="sk">pTM</div><div class="sv2">{short_sum.get('ptm',0):.3f}</div></div>
    </div>
    <div class="sc">
      <div class="sv"><div class="sk">Allele</div><div class="sv2" style="color:#e8947a">{long_name}</div></div>
      <div class="sv"><div class="sk">Ranking</div><div class="sv2">{long_sum['ranking_score']:.3f}</div></div>
      <div class="sv"><div class="sk">ipTM</div><div class="sv2">{long_sum.get('iptm',0):.3f}</div></div>
      <div class="sv"><div class="sk">pTM</div><div class="sv2">{long_sum.get('ptm',0):.3f}</div></div>
    </div>
  </div>
</div>

<div id="viewers">
  <div class="vw">
    <div class="vtitle" style="color:#7aade8">{short_name} — Chain B {short_b_len} aa</div>
    <div class="vbox" id="vs"></div>
  </div>
  <div class="vw">
    <div class="vtitle" style="color:#e8947a">{long_name} — Chain B {long_b_len} aa</div>
    <div class="vbox" id="vl"></div>
  </div>
</div>

<script>
const SCIF = {json.dumps(short_cif)};
const LCIF = {json.dumps(long_cif)};
const L_INS = {json.dumps(long_ins)};
const L_SUB = {json.dumps(long_sub)};
const S_SUB = {json.dumps(short_sub)};

let vS, vL, syncOn = true, syncing = false;

function colorViewer(v, subRes, insRes) {{
  v.setStyle({{chain:'A'}}, {{cartoon:{{color:'#5b8dd9'}}}});
  v.setStyle({{chain:'B'}}, {{cartoon:{{color:'#9b6cbf'}}}});
  if (insRes.length) v.setStyle({{chain:'B', resi:insRes.join(',')}}, {{cartoon:{{color:'#FF6B35'}}}});
  if (subRes.length) v.setStyle({{chain:'B', resi:subRes.join(',')}}, {{cartoon:{{color:'#FFD700'}}}});
  v.render();
}}

function placeViewers() {{
  const topH = document.getElementById('top').offsetHeight;
  document.getElementById('viewers').style.top = topH + 'px';
}}

function init() {{
  placeViewers();

  vS = $3Dmol.createViewer('vs', {{backgroundColor:'#0d1117', antialias:true}});
  vL = $3Dmol.createViewer('vl', {{backgroundColor:'#0d1117', antialias:true}});

  vS.addModel(SCIF, 'cif'); colorViewer(vS, S_SUB, []);   vS.zoomTo(); vS.render();
  vL.addModel(LCIF, 'cif'); colorViewer(vL, L_SUB, L_INS); vL.zoomTo(); vL.render();

  // align starting view: copy short → long after a tick
  requestAnimationFrame(() => {{
    const view = vS.getView();
    vL.setView(view); vL.render();
  }});
}}

function toggleSync() {{
  syncOn = !syncOn;
  const b = document.getElementById('sbtn');
  b.textContent = 'Sync: ' + (syncOn ? 'ON' : 'OFF');
  b.classList.toggle('off', !syncOn);
}}

// sync loop: mirror short → long
let lastV = null;
(function loop() {{
  if (syncOn && vS && vL && !syncing) {{
    const v = JSON.stringify(vS.getView());
    if (v !== lastV) {{ lastV = v; syncing = true; vL.setView(JSON.parse(v)); vL.render(); syncing = false; }}
  }}
  requestAnimationFrame(loop);
}})();

window.addEventListener('DOMContentLoaded', init);
window.addEventListener('resize', () => {{
  placeViewers();
  vS && vS.resize() && vS.render();
  vL && vL.resize() && vL.render();
}});
</script>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    zip_s  = sys.argv[1] if len(sys.argv) > 1 else "fold_run_2_short.zip"
    zip_l  = sys.argv[2] if len(sys.argv) > 2 else "fold_run_4_long.zip"
    out    = sys.argv[3] if len(sys.argv) > 3 else "comparison.html"

    print(f"Loading {zip_s}...")
    sname, sa_s, sb_s, scif, ssum = best_model_cif(zip_s)
    print(f"  {sname}: chain B {len(sb_s)} aa, ranking {ssum['ranking_score']:.3f}")

    print(f"Loading {zip_l}...")
    lname, sa_l, sb_l, lcif, lsum = best_model_cif(zip_l)
    print(f"  {lname}: chain B {len(sb_l)} aa, ranking {lsum['ranking_score']:.3f}")

    print("Superimposing on chain A...")
    lcif = kabsch(scif, lcif)

    print("Computing sequence diff...")
    sa_annot, la_annot, aln_s, aln_l = seq_diff(sb_s, sb_l)
    l_ins = [i for i,a in enumerate(la_annot) if i>0 and a=="ins"]
    l_sub = [i for i,a in enumerate(la_annot) if i>0 and a=="sub"]
    s_sub = [i for i,a in enumerate(sa_annot) if i>0 and a=="sub"]
    print(f"  Extra in long: {l_ins}  |  Sub in long: {l_sub}  |  Sub in short: {s_sub}")

    panel = seq_html(sa_annot, la_annot, aln_s, aln_l)

    print("Writing HTML...")
    html = make_html(sname, lname, len(sb_s), len(sb_l),
                     scif, lcif, ssum, lsum,
                     l_ins, l_sub, s_sub, panel)
    Path(out).write_text(html, encoding="utf-8")
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
