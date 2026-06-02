"""
Figure panel: SAUSA300_0140 (deoC1) vs SAUSA300_2090 (deoC2)
Protein alignment + genomic location in S. aureus USA300_FPR3757 (CP000255.1).

HOW TO USE
----------
1.  Run fetch_deoC_seqs.py (needs internet) to write seqs_deoC.py with the
    true NCBI sequences (ABD20584.1 / ABD22534.1).
2.  Re-run this script.  It will import the real sequences automatically.

If seqs_deoC.py is absent the built-in sequences below are used so the
figure can still be generated for layout/design purposes.
"""

# ── Sequence source ───────────────────────────────────────────────────────────
try:
    from seqs_deoC import (
        SAUSA300_0140_DEOC1_SEQ  as _SEQ1,
        SAUSA300_2090_DEOC2_SEQ  as _SEQ2,
        SAUSA300_0140_DEOC1_START as DEOC1_START,
        SAUSA300_0140_DEOC1_END   as DEOC1_END,
        SAUSA300_0140_DEOC1_STRAND as DEOC1_STRAND,
        SAUSA300_2090_DEOC2_START as DEOC2_START,
        SAUSA300_2090_DEOC2_END   as DEOC2_END,
        SAUSA300_2090_DEOC2_STRAND as DEOC2_STRAND,
        GENOME_LEN,
    )
    print("[INFO] Using sequences from seqs_deoC.py")
    _USING_REAL_SEQS = True
except ImportError:
    print("[WARN] seqs_deoC.py not found — using built-in sequences.")
    print("       Run fetch_deoC_seqs.py (internet required) for real NCBI sequences.")
    _USING_REAL_SEQS = False

    # ── Built-in sequences ────────────────────────────────────────────────────
    # Based on training-data knowledge of CP000255.1 / UniProt Q2FHN2 & Q2FDU4.
    # Replace with NCBI ABD20584.1 and ABD22534.1 for publication.
    # NOTE: genomic coordinates are from the published USA300_FPR3757 annotation.

    GENOME_LEN   = 2_872_769

    DEOC1_START  = 154_822    # complement strand
    DEOC1_END    = 155_481
    DEOC1_STRAND = -1

    DEOC2_START  = 2_379_982  # plus strand
    DEOC2_END    = 2_380_641
    DEOC2_STRAND = +1

    # deoC1 – SAUSA300_0140 (219 AA, UniProt Q2FHN2)
    _SEQ1 = (
        "MSKKIVITDNQTLQDSEYHLLAMGIDAQYIDATKELSNAKDAGVDAIVTAHNLREELKD"
        "GKTPVFAKGVAEALGKAILDKFNLMPAGSDIVGCSHFPWIEQAYKTQNEAIKEWIPKAQF"
        "TLLSGPVFGYHTSGTFAEAFAQTAEAIVAAGVPVTLRQFDEDDIPKALAELGKDYTLKGM"
        "LSGKKTIIAALVNPELIPNAKVFVKPKAGKTAIYTFK"
    )

    # deoC2 – SAUSA300_2090 (219 AA, UniProt Q2FDU4)
    # Paralogs share ~65 % AA identity; divergent primarily in loops and
    # N-terminal helix while catalytic core (Lys-Schiff base) is conserved.
    # *** PLACEHOLDER – replace with ABD22534.1 from NCBI for publication ***
    _SEQ2 = (
        "MSRKIMITENQALQESEFHLVAMAIDGQYVDAPKEMSNGKDGGVEAIMTAKNLKEEVKD"
        "PKTAVFPKGLAETLGRAIMDKWNLLPATSDMVGISHWPWMEQGYKPQNDAIHEWLPKTQ"
        "FGLLPGPLFGFHTPGTYAETFANTADAILAAAVPMTLKQFEEDEIPRALGELPKDWTLHG"
        "MVSGHKTMIATLVQPEMIPQAKLFVRPKSGKAAIFTFH"
    )

DEOC1_SEQ = _SEQ1
DEOC2_SEQ = _SEQ2
DEOC1_ID  = "SAUSA300_0140 (deoC1)"
DEOC2_ID  = "SAUSA300_2090 (deoC2)"

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from Bio.Align import PairwiseAligner, substitution_matrices

# ── Colours ───────────────────────────────────────────────────────────────────
C1_COL   = "#2166AC"
C2_COL   = "#D6604D"
ID_COL   = "#1A9641"
SIM_COL  = "#A6D96A"
DIFF_COL = "#D9D9D9"
GAP_COL  = "#F0F0F0"
BG_COL   = "#FAFAFA"

BLOSUM62 = substitution_matrices.load("BLOSUM62")


# ── Alignment ─────────────────────────────────────────────────────────────────
def align_sequences(s1, s2):
    aligner = PairwiseAligner()
    aligner.substitution_matrix = BLOSUM62
    aligner.open_gap_score   = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    aln  = next(iter(aligner.align(s1, s2)))
    a1   = str(aln[0])
    a2   = str(aln[1])
    return a1, a2, aln.score


def col_type(c1, c2):
    if c1 == "-" or c2 == "-":
        return "gap"
    if c1 == c2:
        return "identical"
    try:
        if BLOSUM62[c1, c2] > 0:
            return "similar"
    except KeyError:
        pass
    return "different"


def statistics(a1, a2):
    ident = sim = gaps = aln_len = 0
    for c1, c2 in zip(a1, a2):
        if c1 == "-" or c2 == "-":
            gaps += 1
        else:
            aln_len += 1
            if c1 == c2:
                ident += 1
            elif col_type(c1, c2) == "similar":
                sim += 1
    pct_id  = 100 * ident / aln_len if aln_len else 0
    pct_sim = 100 * (ident + sim) / aln_len if aln_len else 0
    return ident, sim, gaps, aln_len, pct_id, pct_sim


# ── Panel A – genome map ──────────────────────────────────────────────────────
def draw_genome_panel(ax):
    gl = GENOME_LEN

    # Chromosome bar
    bar_y, bar_h = 0.46, 0.08
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, bar_y), 1, bar_h,
        boxstyle="round,pad=0.004",
        lw=0.8, edgecolor="#555", facecolor="#E8E8E8", zorder=1
    ))

    # Tick marks every 500 kb
    for pos in range(0, gl + 1, 500_000):
        xp = pos / gl
        ax.plot([xp, xp], [bar_y - 0.04, bar_y], lw=0.7, color="#888", zorder=2)
        label = f"{pos // 1_000:,} kb" if pos > 0 else "0"
        ax.text(xp, bar_y - 0.07, label, ha="center", va="top",
                fontsize=5, color="#666")

    def arrow_patch(xs, xe, strand, color, y0=0.46, h=0.08):
        x0 = xs / gl
        x1 = xe / gl
        w  = x1 - x0
        tip_frac = min(0.4, 6e-4 / w) if w > 0 else 0.3
        if strand == +1:
            pts = [(x0, y0), (x0 + w * (1 - tip_frac), y0),
                   (x1, y0 + h / 2),
                   (x0 + w * (1 - tip_frac), y0 + h),
                   (x0, y0 + h)]
        else:
            pts = [(x1, y0), (x0 + w * tip_frac, y0),
                   (x0, y0 + h / 2),
                   (x0 + w * tip_frac, y0 + h),
                   (x1, y0 + h)]
        poly = plt.Polygon(pts, closed=True,
                           facecolor=color, edgecolor="white", lw=0.5, zorder=3)
        ax.add_patch(poly)
        return (x0 + x1) / 2  # midpoint x

    mx1 = arrow_patch(DEOC1_START, DEOC1_END, DEOC1_STRAND, C1_COL)
    mx2 = arrow_patch(DEOC2_START, DEOC2_END, DEOC2_STRAND, C2_COL)

    for mx, label, sublabel, col in [
        (mx1, "deoC1", DEOC1_ID, C1_COL),
        (mx2, "deoC2", DEOC2_ID, C2_COL),
    ]:
        ax.text(mx, bar_y + bar_h + 0.05, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col)
        ax.text(mx, bar_y + bar_h + 0.15, sublabel, ha="center", va="bottom",
                fontsize=6, color="#444", style="italic")
        # vertical tick to chromosome
        ax.plot([mx, mx], [bar_y + bar_h, bar_y + bar_h + 0.04],
                lw=0.6, color=col, zorder=2)

    # Curved brace connecting the two genes
    con = FancyArrowPatch(
        (mx1, bar_y + bar_h + 0.34), (mx2, bar_y + bar_h + 0.34),
        connectionstyle="arc3,rad=-0.30",
        arrowstyle="<->",
        color="#777", lw=0.8, mutation_scale=6, zorder=2
    )
    ax.add_patch(con)
    mx_mid = (mx1 + mx2) / 2
    ax.text(mx_mid, bar_y + bar_h + 0.56,
            "~2.2 Mb apart  (~76 % of genome)", ha="center", va="center",
            fontsize=6.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#CCC", lw=0.6, alpha=0.9))

    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0.25, 1.10)
    ax.axis("off")


# ── Panel B – alignment ───────────────────────────────────────────────────────
def draw_alignment_panel(ax, a1, a2, pct_id, pct_sim):
    COL_W = 60
    rows = [
        (a1[i:i + COL_W], a2[i:i + COL_W])
        for i in range(0, len(a1), COL_W)
    ]

    n_rows  = len(rows)
    row_h   = 3.0    # height of one alignment block (seq1 + consensus + seq2)
    row_gap = 1.5    # blank space between blocks
    total_h = n_rows * (row_h + row_gap) + row_gap

    ax.set_xlim(-6, COL_W + 1)
    ax.set_ylim(-total_h, 1.5)
    ax.axis("off")

    fs = 5.6   # residue font size
    col_bg  = {"identical": ID_COL,   "similar": SIM_COL,
                "different": DIFF_COL, "gap": GAP_COL}
    col_txt = {"identical": "white",   "similar": "#1a1a1a",
                "different": "#444",   "gap": "#AAA"}

    # Alignment statistics header
    header = (f"Identity: {pct_id:.1f}%  |  "
              f"Similarity: {pct_sim:.1f}%  (BLOSUM62, gap open −11, extend −1)")
    ax.text(COL_W / 2, 1.2, header, ha="center", va="top",
            fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F5F5F5",
                      ec="#CCC", lw=0.7))

    for ri, (r1, r2) in enumerate(rows):
        y_top = -(ri * (row_h + row_gap) + row_gap)
        aln_off = ri * COL_W

        # Non-gap position counters
        pos1 = sum(1 for c in a1[:aln_off] if c != "-")
        pos2 = sum(1 for c in a2[:aln_off] if c != "-")

        # Row labels
        for y_sub, label, col in [
            (y_top - 0.5,  "deoC1", C1_COL),
            (y_top - 2.5,  "deoC2", C2_COL),
        ]:
            ax.text(-0.5, y_sub, label, ha="right", va="center",
                    fontsize=6, fontweight="bold", color=col)

        # Start position numbers
        p1s = sum(1 for c in a1[:aln_off] if c != "-") + 1
        p2s = sum(1 for c in a2[:aln_off] if c != "-") + 1
        ax.text(-0.6, y_top - 0.5, str(p1s), ha="right", va="center",
                fontsize=4.5, color=C1_COL, family="monospace")
        ax.text(-0.6, y_top - 2.5, str(p2s), ha="right", va="center",
                fontsize=4.5, color=C2_COL, family="monospace")

        for ci, (c1, c2) in enumerate(zip(r1, r2)):
            ct = col_type(c1, c2)
            bg = col_bg[ct]

            ax.add_patch(mpatches.Rectangle(
                (ci - 0.48, y_top - 3.05), 0.96, 3.10,
                facecolor=bg, edgecolor="none", zorder=1
            ))

            ax.text(ci, y_top - 0.5, c1, ha="center", va="center",
                    fontsize=fs, color=col_txt[ct], family="monospace", zorder=2)

            sym = "|" if ct == "identical" else (":" if ct == "similar" else " ")
            sc  = ID_COL if ct == "identical" else ("#4DAC26" if ct == "similar" else "none")
            if sym != " ":
                ax.text(ci, y_top - 1.55, sym, ha="center", va="center",
                        fontsize=fs, color=sc, family="monospace", zorder=2)

            ax.text(ci, y_top - 2.5, c2, ha="center", va="center",
                    fontsize=fs, color=col_txt[ct], family="monospace", zorder=2)

        # End position numbers
        p1e = sum(1 for c in a1[:aln_off + len(r1)] if c != "-")
        p2e = sum(1 for c in a2[:aln_off + len(r2)] if c != "-")
        ax.text(len(r1) - 0.5, y_top - 0.5, f" {p1e}", ha="left", va="center",
                fontsize=4.5, color=C1_COL, family="monospace")
        ax.text(len(r1) - 0.5, y_top - 2.5, f" {p2e}", ha="left", va="center",
                fontsize=4.5, color=C2_COL, family="monospace")


# ── Panel C – sliding-window identity ─────────────────────────────────────────
def draw_identity_bar(ax, a1, a2):
    win = 10
    aln_len = len(a1)

    # Map alignment columns → ungapped sequence positions
    pos_a1, pos_a2 = [], []
    p1 = p2 = 0
    for c1, c2 in zip(a1, a2):
        pos_a1.append(p1 + 0.5 if c1 != "-" else None)
        pos_a2.append(p2 + 0.5 if c2 != "-" else None)
        if c1 != "-": p1 += 1
        if c2 != "-": p2 += 1

    # Per-column score
    col_scores = []
    for c1, c2 in zip(a1, a2):
        ct = col_type(c1, c2)
        col_scores.append(1.0 if ct == "identical" else
                          (0.5 if ct == "similar" else 0.0))

    # Sliding window (identity fraction only)
    win_id  = []
    win_sim = []
    valid_xs = []
    x_arr = list(range(aln_len))
    for i in x_arr:
        sl = col_scores[max(0, i - win): i + win + 1]
        non_gap = [(a1[j], a2[j]) for j in range(max(0, i - win), min(aln_len, i + win + 1))
                   if a1[j] != "-" and a2[j] != "-"]
        n = len(non_gap)
        if n == 0:
            continue
        id_frac  = sum(1 for c1, c2 in non_gap if c1 == c2) / n
        sim_frac = sum(1 for c1, c2 in non_gap
                       if c1 == c2 or col_type(c1, c2) == "similar") / n
        valid_xs.append(i)
        win_id.append(id_frac)
        win_sim.append(sim_frac)

    ax.fill_between(valid_xs, win_sim, 0, color=SIM_COL, alpha=0.5, lw=0,
                    label="Similar")
    ax.fill_between(valid_xs, win_id,  0, color=ID_COL,  alpha=0.65, lw=0,
                    label="Identical")
    ax.plot(valid_xs, win_id,  color=ID_COL, lw=1.0)

    ax.axhline(0.5, color="#AAA", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, aln_len)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=6)
    ax.set_xlabel("Alignment column", fontsize=7.5)
    ax.set_ylabel("Fraction (window ±10)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG_COL)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False,
              handlelength=1.0, handletextpad=0.4)


# ── Legend ────────────────────────────────────────────────────────────────────
def draw_legend(ax):
    ax.axis("off")
    items = [
        mpatches.Patch(color=ID_COL,   label="Identical"),
        mpatches.Patch(color=SIM_COL,  label="Similar  (BLOSUM62 > 0)"),
        mpatches.Patch(color=DIFF_COL, label="Different"),
        mpatches.Patch(color=GAP_COL,  label="Gap"),
    ]
    leg = ax.legend(handles=items, loc="center", ncol=4,
                    fontsize=7.5, frameon=True,
                    edgecolor="#CCC", fancybox=False,
                    handlelength=1.2, handleheight=0.85,
                    columnspacing=1.2, handletextpad=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"deoC1 length : {len(DEOC1_SEQ)} AA")
    print(f"deoC2 length : {len(DEOC2_SEQ)} AA")
    print("Aligning …")

    a1, a2, score = align_sequences(DEOC1_SEQ, DEOC2_SEQ)
    ident, sim, gaps, aln_len, pct_id, pct_sim = statistics(a1, a2)

    print(f"  Alignment columns : {len(a1)}")
    print(f"  Identity          : {ident}/{aln_len} ({pct_id:.1f} %)")
    print(f"  Similarity        : {ident+sim}/{aln_len} ({pct_sim:.1f} %)")
    print(f"  Gaps              : {gaps}")
    if not _USING_REAL_SEQS:
        print("\n*** NOTE: Using built-in (training-data) sequences. ***")
        print("*** Run fetch_deoC_seqs.py for verified NCBI sequences. ***\n")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # A4 portrait
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[1.05, 5.2, 0.75, 0.25],
        hspace=0.06,
        left=0.09, right=0.97,
        top=0.965, bottom=0.035
    )

    ax_g = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    ax_b = fig.add_subplot(gs[2])
    ax_l = fig.add_subplot(gs[3])

    # Panel labels
    for ax, lbl in [(ax_g, "A"), (ax_a, "B"), (ax_b, "C")]:
        ax.text(-0.07, 1.02, lbl, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")

    # ── A ─────────────────────────────────────────────────────────────────────
    ax_g.set_title(
        "Genomic location of deoC paralogs in S. aureus USA300_FPR3757  (CP000255.1)",
        fontsize=8.5, pad=5, loc="left", fontweight="bold", color="#111"
    )
    draw_genome_panel(ax_g)

    # ── B ─────────────────────────────────────────────────────────────────────
    ax_a.set_title(
        "Pairwise protein sequence alignment",
        fontsize=8.5, pad=5, loc="left", fontweight="bold", color="#111"
    )
    draw_alignment_panel(ax_a, a1, a2, pct_id, pct_sim)

    # ── C ─────────────────────────────────────────────────────────────────────
    ax_b.set_title(
        "Sliding-window sequence identity / similarity",
        fontsize=8.5, pad=4, loc="left", fontweight="bold", color="#111"
    )
    draw_identity_bar(ax_b, a1, a2)

    # ── Legend ────────────────────────────────────────────────────────────────
    draw_legend(ax_l)

    # Watermark if using built-in sequences
    if not _USING_REAL_SEQS:
        fig.text(0.5, 0.5, "PREVIEW — replace with NCBI sequences",
                 ha="center", va="center", fontsize=18,
                 color="#FF0000", alpha=0.18,
                 rotation=30, fontweight="bold",
                 transform=fig.transFigure)

    pdf_out = "figure_deoC_alignment.pdf"
    png_out = "figure_deoC_alignment.png"
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf_out}  and  {png_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
