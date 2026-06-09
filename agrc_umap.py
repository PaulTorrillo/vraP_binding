import os
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load agrC typing from supplementary table
df = pd.read_excel("DatasetS1.xlsx", sheet_name="TableS3")
acc_to_agr = dict(zip(df["Accession"], df["agr group"]))

# Load all .pt embeddings
pt_dir = "agrC_fresh"
pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])

embeddings = []
labels = []

for fname in pt_files:
    accession = fname.split("_")[0]
    data = torch.load(os.path.join(pt_dir, fname), map_location="cpu")
    vec = data["mean_representations"][6].numpy()
    embeddings.append(vec)
    labels.append(acc_to_agr.get(accession, "unknown"))

embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
print("Label counts:", pd.Series(labels).value_counts().to_string())

# Run UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(embeddings)

# Plot
agr_colors = {
    "gp1": "#E41A1C",
    "gp2": "#377EB8",
    "gp3": "#4DAF4A",
    "gp4": "#FF7F00",
    "unknown": "#999999",
}

fig, ax = plt.subplots(figsize=(8, 6))

for group, color in agr_colors.items():
    mask = labels == group
    if mask.sum() == 0:
        continue
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=color,
        label=group,
        s=25,
        alpha=0.8,
        linewidths=0,
    )

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of agrC protein embeddings colored by agr group")
ax.legend(title="agr group", frameon=True, loc="best")
plt.tight_layout()
plt.savefig("agrc_umap.png", dpi=150)
print("Saved agrc_umap.png")
