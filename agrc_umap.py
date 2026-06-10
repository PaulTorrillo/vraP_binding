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

lengths    = np.array(list(seq_lengths.values()))
names      = list(seq_lengths.keys())
mean_len   = lengths.mean()
std_len    = lengths.std()
median_len = np.median(lengths)

# ── GMM: identify full-length vs truncated populations ───────────────────────
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(lengths.reshape(-1, 1))
component_labels      = gmm.predict(lengths.reshape(-1, 1))
full_length_component = int(np.argmax(gmm.means_))
component_means       = gmm.means_.flatten()
component_stds        = np.sqrt(gmm.covariances_).flatten()

n_full     = int((component_labels == full_length_component).sum())
n_truncated = int((component_labels != full_length_component).sum())

print(f"Sequence length  mean: {mean_len:.1f}  median: {median_len:.1f}  std: {std_len:.1f}")
print(f"\nGMM components:")
for c in range(2):
    tag = "full-length" if c == full_length_component else "truncated"
    print(f"  [{tag}]: mean={component_means[c]:.1f} aa, "
          f"std={component_stds[c]:.1f} aa, n={int((component_labels==c).sum())}")

# Map each sequence name to its GMM label
name_to_gmm = {names[i]: component_labels[i] for i in range(len(names))}

# ── Load ALL embeddings ───────────────────────────────────────────────────────
pt_dir = "agrC_fresh"
embeddings, gmm_labels, basenames = [], [], []

for fname in sorted(os.listdir(pt_dir)):
    if not fname.endswith(".pt"):
        continue
    label = fname[:-3]
    if label not in name_to_gmm:
        continue
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    embeddings.append(data["mean_representations"][6].numpy())
    gmm_labels.append(name_to_gmm[label])
    basenames.append(fname)

embeddings = np.array(embeddings)
gmm_labels = np.array(gmm_labels)
print(f"\nTotal embeddings loaded: {len(embeddings)}")

# ── Load agrC typing for right panel ─────────────────────────────────────────
df_meta   = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df_meta["Accession"], df_meta["agr group"]))
agr_labels = np.array([acc_to_agr.get(f.split("_")[0], "unknown") for f in basenames])

# ── PCA on all embeddings ─────────────────────────────────────────────────────
pca_all  = PCA(n_components=2, random_state=42)
coords_all = pca_all.fit_transform(embeddings)
var_all    = pca_all.explained_variance_ratio_ * 100
print(f"PCA (all):          PC1: {var_all[0]:.1f}%  PC2: {var_all[1]:.1f}%")

# ── PCA on full-length embeddings only ───────────────────────────────────────
fl_mask      = gmm_labels == full_length_component
pca_fl       = PCA(n_components=2, random_state=42)
coords_fl    = pca_fl.fit_transform(embeddings[fl_mask])
var_fl       = pca_fl.explained_variance_ratio_ * 100
agr_fl       = agr_labels[fl_mask]
print(f"PCA (full-length):  PC1: {var_fl[0]:.1f}%  PC2: {var_fl[1]:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
# Tableau 10 split across panels — zero color overlap, publication-ready
gmm_colors = {
    "full-length": "#4E79A7",   # steel blue
    "truncated":   "#F28E2B",   # amber orange
}
agr_colors = {
    "gp1":     "#E15759",   # coral red
    "gp2":     "#76B7B2",   # teal
    "gp3":     "#59A14F",   # green
    "gp4":     "#B07AA1",   # muted purple
    "unknown": "#BAB0AC",   # warm grey
}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ── Left panel: all embeddings, colored by GMM population ────────────────────
ax = axes[0]
groups = [
    (full_length_component,     "full-length", gmm_colors["full-length"], 25, 0.8),
    (1 - full_length_component, "truncated",   gmm_colors["truncated"],   40, 0.9),
]
for comp, label, color, size, alpha in groups:
    mask = gmm_labels == comp
    c    = comp
    legend_label = (f"{label} (n={mask.sum()}, "
                    f"μ={component_means[c]:.1f} aa, "
                    f"σ={component_stds[c]:.1f} aa)")
    ax.scatter(coords_all[mask, 0], coords_all[mask, 1],
               c=color, label=legend_label, s=size, alpha=alpha, linewidths=0)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
ax.set_xlabel(f"PC1 — all ({var_all[0]:.1f}%)")
ax.set_ylabel(f"PC2 — all ({var_all[1]:.1f}%)")
ax.set_title("Gaussian Mixture Model length classification")
ax.legend(title="Population", frameon=True, fontsize=8, loc="upper left")

# ── Right panel: full-length only, colored by agr group ──────────────────────
ax = axes[1]
for group, color in agr_colors.items():
    mask = agr_fl == group
    if mask.sum() == 0:
        continue
    ax.scatter(coords_fl[mask, 0], coords_fl[mask, 1],
               c=color, label=f"{group} (n={mask.sum()})",
               s=25, alpha=0.8, linewidths=0)

ax.set_xlabel(f"PC1 — full-length ({var_fl[0]:.1f}%)")
ax.set_ylabel(f"PC2 — full-length ({var_fl[1]:.1f}%)")
ax.set_title("Full-length embeddings — agr group")
ax.legend(title="agr group", frameon=True, fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("agrc_pca.png", dpi=300)
print("Saved agrc_pca.png")
