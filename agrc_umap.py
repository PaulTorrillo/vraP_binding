import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# ── Parse sequence lengths from FASTA ────────────────────────────────────────
seq_lengths = {}
with open("final_genes.faa") as fh:
    name, seq = None, []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if name:
                seq_lengths[name] = len("".join(seq))
            name = line[1:]
            seq = []
        else:
            seq.append(line)
    if name:
        seq_lengths[name] = len("".join(seq))

fasta_names = list(seq_lengths.keys())
lengths     = np.array(list(seq_lengths.values()))

# ── GMM: full-length vs truncated ────────────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(lengths.reshape(-1, 1))
gmm_comp         = gmm.predict(lengths.reshape(-1, 1))
full_comp        = int(np.argmax(gmm.means_))
name_to_is_trunc = {fasta_names[i]: (gmm_comp[i] != full_comp)
                    for i in range(len(fasta_names))}
name_to_len      = dict(zip(fasta_names, lengths))

# ── Load metadata ─────────────────────────────────────────────────────────────
df_meta    = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df_meta["Accession"], df_meta["agr group"]))

df_tsv = pd.read_csv("pt_cluster_labels.tsv", sep="\t")
df_tsv["label"] = df_tsv["basename"].str.strip().str.replace(".pt", "", regex=False)
label_to_assigned = dict(zip(df_tsv["label"], df_tsv["assigned"]))
label_to_cluster  = dict(zip(df_tsv["label"], df_tsv["cluster_label"]))

# ── Load all embeddings ───────────────────────────────────────────────────────
pt_dir    = "agrC_fresh"
rows      = []

for fname in sorted(os.listdir(pt_dir)):
    if not fname.endswith(".pt"):
        continue
    label     = fname[:-3]
    accession = fname.split("_")[0]
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    vec  = data["mean_representations"][6].numpy()

    p1_group = ("truncated" if name_to_is_trunc.get(label, False)
                else acc_to_agr.get(accession, "unknown"))
    assigned  = label_to_assigned.get(label, 0)
    locus_tag = (label_to_cluster.get(label, "").split()[0]
                 if assigned == 1 else "unclustered")

    rows.append({"label": label, "vec": vec,
                 "p1_group": p1_group, "locus_tag": locus_tag,
                 "length": name_to_len.get(label, np.nan)})

embeddings = np.array([r["vec"] for r in rows])
df_pts     = pd.DataFrame([{k: v for k, v in r.items() if k != "vec"} for r in rows])
print(f"Loaded {len(embeddings)} embeddings")

# ── PCA on all embeddings ─────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
var    = pca.explained_variance_ratio_ * 100
df_pts["pc1"] = coords[:, 0]
df_pts["pc2"] = coords[:, 1]
print(f"PC1: {var[0]:.1f}%  PC2: {var[1]:.1f}%")

xlim = (-3, 3)
ylim = (-3, 3)

# ── Colour maps ───────────────────────────────────────────────────────────────
# ── Colour maps ───────────────────────────────────────────────────────────────
# Blue family  → FEOBHI_09345 (main AgrC) / gp1 / gp4
# Red family   → GFNPHG_09685 (AgrC HK)  / gp3
# Green family → HCCNHA_05915 (YesM)     / gp2
# Dark grey    → truncated / unclustered / unknown

p1_colors = {
    "gp1":       "#4393C3",   # cornflower blue   (blue family)
    "gp4":       "#6A3D9A",   # vivid purple
    "gp3":       "#D6604D",   # brick red         (red family)
    "gp2":       "#91CF60",   # lime green        (green family)
    "unknown":   "#CC7722",   # warm ochre        (clearly distinct from all greys and blues)
    "truncated": "#525252",   # dark grey
}

locus_color = {
    "FEOBHI_09345": "#6BAED6",   # light blue        (blue family — main AgrC)
    "GFNPHG_09685": "#F4A582",   # salmon            (red family  — AgrC HK)
    "HCCNHA_05915": "#1A9850",   # forest green      (green family — YesM)
    "FEOBHI_10025": "#762A83",   # purple            (other)
    "FEOBHI_00575": "#E7298A",   # hot pink          (other)
    "FEOBHI_06835": "#A65628",   # brown             (other)
    "FEOBHI_12555": "#FF7F00",   # amber             (other)
    "MDBBCG_11375": "#E6AB02",   # golden yellow     (other)
}
unique_loci = sorted(df_pts.loc[df_pts["locus_tag"] != "unclustered", "locus_tag"].unique())

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Panel 1: agr group / truncated ───────────────────────────────────────────
ax = axes[0]
# background groups first (low z-order)
for group, zord in [("truncated", 1), ("unknown", 1),
                    ("gp1", 2), ("gp2", 2), ("gp3", 2), ("gp4", 2)]:
    sub = df_pts[df_pts["p1_group"] == group]
    if sub.empty:
        continue
    lo, hi = int(sub["length"].min()), int(sub["length"].max())
    lbl = f"{group}  n={len(sub)}, {lo}–{hi} aa"
    ax.scatter(sub["pc1"], sub["pc2"],
               c=p1_colors[group], label=lbl,
               s=20, alpha=0.8, linewidths=0, zorder=zord)

ax.set_xscale("symlog", linthresh=0.2)
ax.set_yscale("symlog", linthresh=0.2)
ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("agrC embeddings — agr group / truncated")
ax.legend(title="Group", frameon=True, fontsize=7.5, loc="best",
          handlelength=1, borderpad=0.7)

# ── Panel 2: cluster locus tag / unclustered ──────────────────────────────────
ax = axes[1]
unc = df_pts[df_pts["locus_tag"] == "unclustered"]
ax.scatter(unc["pc1"], unc["pc2"],
           c="#525252", s=15, alpha=0.6, linewidths=0,
           label=f"unclustered  n={len(unc)}", zorder=1)

for locus in unique_loci:
    sub = df_pts[df_pts["locus_tag"] == locus]
    lo, hi = int(sub["length"].min()), int(sub["length"].max())
    lbl = f"{locus}  n={len(sub)}, {lo}–{hi} aa"
    ax.scatter(sub["pc1"], sub["pc2"],
               c=locus_color.get(locus, "#AAAAAA"), label=lbl,
               s=20, alpha=0.85, linewidths=0, zorder=2)

ax.set_xscale("symlog", linthresh=0.2)
ax.set_yscale("symlog", linthresh=0.2)
ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("agrC embeddings — cluster (locus tag)")
ax.legend(title="Cluster", frameon=True, fontsize=7, loc="best",
          handlelength=1, borderpad=0.7)

plt.tight_layout()
plt.savefig("agrc_pca.png", dpi=300)
print("Saved agrc_pca.png")
