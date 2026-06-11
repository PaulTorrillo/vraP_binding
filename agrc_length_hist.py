import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture

# ── Parse sequence lengths ────────────────────────────────────────────────────
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

names   = list(seq_lengths.keys())
lengths = np.array(list(seq_lengths.values()))

# ── GMM ───────────────────────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(lengths.reshape(-1, 1))
labels = gmm.predict(lengths.reshape(-1, 1))
full_component = int(np.argmax(gmm.means_))
trunc_component = 1 - full_component

fl_lengths = lengths[labels == full_component]
tr_lengths = lengths[labels == trunc_component]

means = gmm.means_.flatten()
stds  = np.sqrt(gmm.covariances_).flatten()

# ── Load unassigned info ──────────────────────────────────────────────────────
df_tsv = pd.read_csv("pt_cluster_labels.tsv", sep="\t")
df_tsv["basename"] = df_tsv["basename"].str.strip()
unassigned_names = set(
    df_tsv.loc[df_tsv["assigned"] == 0, "basename"]
    .str.replace(".pt", "", regex=False)
)
ua_lengths = np.array([seq_lengths[n] for n in names if n in unassigned_names])

# ── Plot ──────────────────────────────────────────────────────────────────────
# Use log-spaced bins to spread out the dense 427/430 bars from the sparse tail
bins = np.linspace(0, 460, 47)   # 10 aa bins

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(fl_lengths, bins=bins, color="#4E79A7", alpha=0.85,
        label=f"full-length  (n={len(fl_lengths)}, μ={means[full_component]:.1f} aa, σ={stds[full_component]:.1f} aa)",
        zorder=2)
ax.hist(tr_lengths, bins=bins, color="#F28E2B", alpha=0.85,
        label=f"truncated    (n={len(tr_lengths)}, μ={means[trunc_component]:.1f} aa, σ={stds[trunc_component]:.1f} aa)",
        zorder=3)

# Rug marks for unassigned sequences along the x-axis
ax.plot(ua_lengths, np.full_like(ua_lengths, -0.6, dtype=float),
        '|', color="#E15759", markersize=10, markeredgewidth=1.5,
        label=f"unassigned to cluster (n={len(ua_lengths)})", zorder=4, clip_on=False)

ax.set_xlabel("Sequence length (aa)")
ax.set_ylabel("Count")
ax.set_title("agrC sequence length distribution — Gaussian Mixture Model groupings")
ax.set_xlim(0, 460)
ax.set_ylim(-1.5)
ax.legend(frameon=True, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("agrc_length_hist.png", dpi=300)
print("Saved agrc_length_hist.png")
