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

xlim = (-1.8, 2.2)
ylim = (-1.7, 0.4)

# ── Colour maps ───────────────────────────────────────────────────────────────
p1_colors = {
    "gp1":       "#E15759",
    "gp2":       "#76B7B2",
    "gp3":       "#59A14F",
    "gp4":       "#B07AA1",
    "unknown":   "#BAB0AC",
    "truncated": "#F28E2B",
}

cluster_palette = [
    "#4E79A7", "#E15759", "#59A14F", "#F28E2B",
    "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7",
]
unique_loci = sorted(df_pts.loc[df_pts["locus_tag"] != "unclustered", "locus_tag"].unique())
locus_color = {loc: cluster_palette[i % len(cluster_palette)]
               for i, loc in enumerate(unique_loci)}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Panel 1: agr group / truncated ───────────────────────────────────────────
ax = axes[0]
for group in ["gp1", "gp2", "gp3", "gp4", "unknown", "truncated"]:
    sub = df_pts[df_pts["p1_group"] == group]
    if sub.empty:
        continue
    lo, hi = int(sub["length"].min()), int(sub["length"].max())
    lbl = f"{group}  n={len(sub)}, {lo}–{hi} aa"
    ax.scatter(sub["pc1"], sub["pc2"],
               c=p1_colors[group], label=lbl,
               s=20, alpha=0.8, linewidths=0, zorder=2)

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
           c="black", s=15, alpha=0.6, linewidths=0,
           label=f"unclustered  n={len(unc)}", zorder=1)

for locus in unique_loci:
    sub = df_pts[df_pts["locus_tag"] == locus]
    lo, hi = int(sub["length"].min()), int(sub["length"].max())
    lbl = f"{locus}  n={len(sub)}, {lo}–{hi} aa"
    ax.scatter(sub["pc1"], sub["pc2"],
               c=locus_color[locus], label=lbl,
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
