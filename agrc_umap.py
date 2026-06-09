import os
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# ── Load agrC typing from supplementary table ─────────────────────────────────
df_meta = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df_meta["Accession"], df_meta["agr group"]))

# ── Load cluster assignments from TSV ─────────────────────────────────────────
df_tsv = pd.read_csv("pt_cluster_labels.tsv", sep="\t")
df_tsv["basename"] = df_tsv["basename"].str.strip()
basename_to_cluster = dict(zip(df_tsv["basename"], df_tsv["cluster_label"]))
basename_to_assigned = dict(zip(df_tsv["basename"], df_tsv["assigned"]))

# ── Load all .pt embeddings ───────────────────────────────────────────────────
pt_dir = "agrC_fresh"
pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])

embeddings, agr_labels, cluster_labels, assigned_flags = [], [], [], []

for fname in pt_files:
    accession = fname.split("_")[0]
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    embeddings.append(data["mean_representations"][6].numpy())
    agr_labels.append(acc_to_agr.get(accession, "unknown"))
    cluster_labels.append(basename_to_cluster.get(fname, "unassigned"))
    assigned_flags.append(int(basename_to_assigned.get(fname, 0)))

embeddings    = np.array(embeddings)
agr_labels    = np.array(agr_labels)
cluster_labels = np.array(cluster_labels)
assigned_flags = np.array(assigned_flags)

print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
print("agr group counts:\n", pd.Series(agr_labels).value_counts().to_string())
print("cluster counts:\n", pd.Series(cluster_labels).value_counts().to_string())

# ── UMAP ──────────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(embeddings)

# ── Colour palettes ───────────────────────────────────────────────────────────
agr_colors = {
    "gp1":     "#E41A1C",
    "gp2":     "#377EB8",
    "gp3":     "#4DAF4A",
    "gp4":     "#FF7F00",
    "unknown": "#999999",
}

# Short display names for the 8 cluster labels
cluster_short = {
    "FEOBHI_09345 Accessory gene regulator C":                                  "AgrC (FEOBHI_09345)",
    "GFNPHG_09685 Two-component system AgrA family sensor histidine kinase AgrC": "AgrC HK (GFNPHG_09685)",
    "HCCNHA_05915 Sensor histidine kinase YesM":                                "HK YesM (HCCNHA_05915)",
    "FEOBHI_06835 histidine kinase GraS/ApsS":                                  "GraS/ApsS (FEOBHI_06835)",
    "FEOBHI_10025 anti-sigma B factor RsbW":                                    "RsbW (FEOBHI_10025)",
    "FEOBHI_00575 DUF5080 domain-containing protein":                           "DUF5080 (FEOBHI_00575)",
    "FEOBHI_12555 AAA family ATPase":                                           "AAA ATPase (FEOBHI_12555)",
    "MDBBCG_11375 DUF3169 domain-containing protein":                           "DUF3169 (MDBBCG_11375)",
}

# Distinct colours for 8 clusters (ColorBrewer Set1 + extras)
cluster_colors = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00",
    "#984EA3", "#A65628", "#F781BF", "#00CED1",
]
unique_clusters = sorted(
    [c for c in np.unique(cluster_labels) if c != "unassigned"],
    key=lambda x: cluster_short.get(x, x),
)
cluster_color_map = {c: cluster_colors[i] for i, c in enumerate(unique_clusters)}

# ── Two-panel figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left panel: agr group ─────────────────────────────────────────────────────
ax = axes[0]
for group, color in agr_colors.items():
    mask = agr_labels == group
    if mask.sum() == 0:
        continue
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, label=group, s=25, alpha=0.8, linewidths=0)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("agrC embeddings — agr group")
ax.legend(title="agr group", frameon=True, fontsize=9)

# ── Right panel: cluster assignment ──────────────────────────────────────────
ax = axes[1]

# Draw unassigned as faint grey background points first
unassigned_mask = assigned_flags == 0
ax.scatter(coords[unassigned_mask, 0], coords[unassigned_mask, 1],
           c="#DDDDDD", s=18, alpha=0.4, linewidths=0, zorder=1)

# Draw assigned points coloured by cluster on top
for cluster_label in unique_clusters:
    mask = (cluster_labels == cluster_label) & (assigned_flags == 1)
    if mask.sum() == 0:
        continue
    color = cluster_color_map[cluster_label]
    short = cluster_short.get(cluster_label, cluster_label)
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, label=short, s=30, alpha=0.85, linewidths=0, zorder=2)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("agrC embeddings — cluster assignment")
ax.legend(title="Cluster", frameon=True, fontsize=7, loc="best",
          handlelength=1.2, borderpad=0.8)

plt.tight_layout()
plt.savefig("agrc_umap.png", dpi=150)
print("Saved agrc_umap.png")
