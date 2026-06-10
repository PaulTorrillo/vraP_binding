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

# ── PCA on all embeddings ─────────────────────────────────────────────────────
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
var    = pca.explained_variance_ratio_ * 100
print(f"PC1: {var[0]:.1f}%  PC2: {var[1]:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

groups = [
    (full_length_component, "full-length", "#377EB8", 25, 0.8),
    (1 - full_length_component, "truncated",    "#E41A1C", 40, 0.9),
]
for comp, label, color, size, alpha in groups:
    mask = gmm_labels == comp
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, label=f"{label} (n={mask.sum()})",
               s=size, alpha=alpha, linewidths=0)

ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("agrC embeddings — GMM length classification")
ax.legend(title="Population", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("agrc_pca.png", dpi=150)
print("Saved agrc_pca.png")
