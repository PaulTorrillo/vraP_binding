import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# ── Parse sequences from FASTA ────────────────────────────────────────────────
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

lengths = np.array(list(seq_lengths.values()))
mean_len   = lengths.mean()
std_len    = lengths.std()
median_len = np.median(lengths)

# Fit a 2-component GMM to separate the full-length cluster from truncated sequences.
# IQR/MAD collapse to zero here because 357/385 sequences are exactly 427 or 430 aa,
# making threshold-based methods degenerate. GMM models both populations directly and
# assigns each sequence to the component whose mean is higher (full-length cluster).
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(lengths.reshape(-1, 1))
component_labels = gmm.predict(lengths.reshape(-1, 1))
full_length_component = int(np.argmax(gmm.means_))
component_means = gmm.means_.flatten()
component_stds  = np.sqrt(gmm.covariances_).flatten()

names = list(seq_lengths.keys())
kept    = {names[i]: lengths[i] for i in range(len(names))
           if component_labels[i] == full_length_component}
removed = len(seq_lengths) - len(kept)

print(f"Sequence length  mean: {mean_len:.1f}  median: {median_len:.1f}  std: {std_len:.1f}")
print(f"\nGMM components:")
for c in range(2):
    tag = "full-length (kept)" if c == full_length_component else "truncated (removed)"
    print(f"  Component {c} [{tag}]: mean={component_means[c]:.1f} aa, "
          f"std={component_stds[c]:.1f} aa, n={int((component_labels==c).sum())}")
print(f"\nTotal: {len(seq_lengths)}  kept: {len(kept)}  removed: {removed}")

# ── Load agrC typing ──────────────────────────────────────────────────────────
df_meta = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df_meta["Accession"], df_meta["agr group"]))

# ── Load embeddings for sequences passing the length filter ──────────────────
pt_dir = "agrC_fresh"
embeddings, agr_labels, basenames = [], [], []

for fname in sorted(os.listdir(pt_dir)):
    if not fname.endswith(".pt"):
        continue
    label = fname[:-3]          # strip .pt → matches FASTA header
    if label not in kept:
        continue
    accession = fname.split("_")[0]
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    embeddings.append(data["mean_representations"][6].numpy())
    agr_labels.append(acc_to_agr.get(accession, "unknown"))
    basenames.append(fname)

embeddings = np.array(embeddings)
agr_labels = np.array(agr_labels)
print(f"\nEmbeddings loaded for length-filtered set: {len(embeddings)}")
print("agr group counts:\n", pd.Series(agr_labels).value_counts().to_string())

# ── PCA on all length-filtered sequences ─────────────────────────────────────
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
var    = pca.explained_variance_ratio_ * 100
print(f"\nPC1: {var[0]:.1f}%  PC2: {var[1]:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
agr_colors = {
    "gp1":     "#E41A1C",
    "gp2":     "#377EB8",
    "gp3":     "#4DAF4A",
    "gp4":     "#FF7F00",
    "unknown": "#999999",
}

fig, ax = plt.subplots(figsize=(8, 6))
for group, color in agr_colors.items():
    mask = agr_labels == group
    if mask.sum() == 0:
        continue
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, label=group, s=25, alpha=0.8, linewidths=0)

ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("agrC embeddings — agr group (GMM length filter)")
ax.legend(title="agr group", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("agrc_pca.png", dpi=150)
print("Saved agrc_pca.png")
