import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Liberation Sans"],
    "font.size":             15,
    "axes.titlesize":        17,
    "axes.labelsize":        16,
    "xtick.labelsize":       14,
    "ytick.labelsize":       14,
    "legend.fontsize":       10,
    "legend.title_fontsize": 11,
})

# ── Parse sequence lengths ────────────────────────────────────────────────────
seq_lengths = {}
with open("final_genes.faa") as fh:
    name, seq = None, []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if name:
                seq_lengths[name] = len("".join(seq))
            name = line[1:]; seq = []
        else:
            seq.append(line)
    if name:
        seq_lengths[name] = len("".join(seq))

fasta_names = list(seq_lengths.keys())
lengths     = np.array(list(seq_lengths.values()))

# ── GMM ───────────────────────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(lengths.reshape(-1, 1))
gmm_comp         = gmm.predict(lengths.reshape(-1, 1))
full_comp        = int(np.argmax(gmm.means_))
name_to_is_trunc = {fasta_names[i]: (gmm_comp[i] != full_comp)
                    for i in range(len(fasta_names))}
name_to_len      = dict(zip(fasta_names, lengths))

# ── Metadata ──────────────────────────────────────────────────────────────────
df_meta    = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df_meta["Accession"], df_meta["agr group"]))

df_tsv = pd.read_csv("pt_cluster_labels.tsv", sep="\t")
df_tsv["label"]  = df_tsv["basename"].str.strip().str.replace(".pt", "", regex=False)
df_tsv["locus"]  = df_tsv["cluster_label"].str.split().str[0]
label_to_assigned = dict(zip(df_tsv["label"], df_tsv["assigned"]))
label_to_locus    = dict(zip(df_tsv["label"], df_tsv["locus"]))

# ESM-2 cluster numbering (FEO → 1, HCC → 2, GFN → 3)
esm_cluster_order = ["FEOBHI_09345", "HCCNHA_05915", "GFNPHG_09685"]
locus_to_cluster  = {loc: f"ESM-2 cluster {i+1}"
                     for i, loc in enumerate(esm_cluster_order)}

# ── Load all embeddings ───────────────────────────────────────────────────────
pt_dir = "agrC_fresh"
rows   = []

for fname in sorted(os.listdir(pt_dir)):
    if not fname.endswith(".pt"):
        continue
    label     = fname[:-3]
    accession = fname.split("_")[0]
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    vec  = data["mean_representations"][6].numpy()

    agr_group = ("truncated" if name_to_is_trunc.get(label, False)
                 else acc_to_agr.get(accession, "unknown"))
    assigned  = label_to_assigned.get(label, 0)
    locus     = label_to_locus.get(label, "")
    cluster   = locus_to_cluster.get(locus, "unclustered") if assigned == 1 else "unclustered"

    rows.append({"label": label, "vec": vec, "agr_group": agr_group,
                 "cluster": cluster, "length": name_to_len.get(label, np.nan)})

embeddings = np.array([r["vec"] for r in rows])
df_pts     = pd.DataFrame([{k: v for k, v in r.items() if k != "vec"} for r in rows])

# ── PCA ───────────────────────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
var    = pca.explained_variance_ratio_ * 100
df_pts["pc1"] = coords[:, 0]
df_pts["pc2"] = coords[:, 1]
print(f"PC1: {var[0]:.1f}%  PC2: {var[1]:.1f}%")

# ── Tight axis limits ─────────────────────────────────────────────────────────
pad  = 0.04
xspan = coords[:, 0].max() - coords[:, 0].min()
yspan = coords[:, 1].max() - coords[:, 1].min()
xlim = (coords[:, 0].min() - pad * xspan, coords[:, 0].max() + pad * xspan)
ylim = (coords[:, 1].min() - pad * yspan, coords[:, 1].max() + pad * yspan)

# ── Colour + shape maps ───────────────────────────────────────────────────────
p1_colors = {
    "gp1":       "#4393C3",
    "gp2":       "#91CF60",
    "gp3":       "#D6604D",
    "gp4":       "#6A3D9A",
    "unknown":   "#CC7722",
    "truncated": "black",
}
cluster_markers = {
    "ESM-2 cluster 1": "o",
    "ESM-2 cluster 2": "s",
    "ESM-2 cluster 3": "^",
    "unclustered":     "x",
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.subplots_adjust(right=0.62)   # leave room for legends on the right

agr_order     = ["gp1", "gp2", "gp3", "gp4", "unknown", "truncated"]
cluster_order = ["ESM-2 cluster 1", "ESM-2 cluster 2", "ESM-2 cluster 3", "unclustered"]

# draw truncated/unknown at low z-order, rest on top
zorder_map = {"truncated": 1, "unknown": 1}

for agr in agr_order:
    sub = df_pts[df_pts["agr_group"] == agr]
    if sub.empty:
        continue
    zord = zorder_map.get(agr, 2)
    for clust in cluster_order:
        subsub = sub[sub["cluster"] == clust]
        if subsub.empty:
            continue
        ax.scatter(subsub["pc1"], subsub["pc2"],
                   c=p1_colors[agr],
                   marker=cluster_markers[clust],
                   s=55, alpha=0.55,
                   linewidths=1.8 if clust == "unclustered" else 0.4,
                   edgecolors="none" if clust != "unclustered" else p1_colors[agr],
                   zorder=zord)

ax.set_xscale("symlog", linthresh=0.2)
ax.set_yscale("symlog", linthresh=0.2)
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("agrC ESM-2 embeddings — PCA")

# ── Legend 1: agr group (colors) ─────────────────────────────────────────────
color_handles = [
    mpatches.Patch(color=p1_colors[g], label=g)
    for g in agr_order if not df_pts[df_pts["agr_group"] == g].empty
]
leg1 = ax.legend(handles=color_handles, title="agr group",
                 loc="upper left", bbox_to_anchor=(1.02, 1.0),
                 frameon=True, borderpad=0.7, handlelength=1.2)
ax.add_artist(leg1)

# ── Legend 2: ESM-2 cluster (shapes) ─────────────────────────────────────────
shape_handles = [
    mlines.Line2D([0], [0], marker=cluster_markers[c], color="#444444",
                  linestyle="None", markersize=8,
                  markeredgewidth=0.4 if c != "unclustered" else 2.0,
                  label=c)
    for c in cluster_order
]
ax.legend(handles=shape_handles, title="ESM-2 cluster",
          loc="upper left", bbox_to_anchor=(1.02, 0.42),
          frameon=True, borderpad=0.7, handlelength=1.2)

plt.savefig("agrc_pca_single.png", dpi=300, bbox_inches="tight")
print("Saved agrc_pca_single.png")
