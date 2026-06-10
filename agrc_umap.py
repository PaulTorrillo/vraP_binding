import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

cutoff  = median_len * 0.90
kept    = {k: v for k, v in seq_lengths.items() if v >= cutoff}
removed = len(seq_lengths) - len(kept)

print(f"Sequence length  mean: {mean_len:.1f}  median: {median_len:.1f}  std: {std_len:.1f}")
print(f"90% of median cutoff: {cutoff:.1f} aa")
print(f"Total: {len(seq_lengths)}  kept: {len(kept)}  removed (<cutoff): {removed}")

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
ax.set_title("agrC embeddings — agr group (length ≥ 90% of median)")
ax.legend(title="agr group", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("agrc_pca.png", dpi=150)
print("Saved agrc_pca.png")
