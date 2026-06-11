import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ── Load cluster assignment ───────────────────────────────────────────────────
df_tsv = pd.read_csv("pt_cluster_labels.tsv", sep="\t")
df_tsv["basename"] = df_tsv["basename"].str.strip()
df_tsv["label"] = df_tsv["basename"].str.replace(".pt", "", regex=False)
assigned_names   = set(df_tsv.loc[df_tsv["assigned"] == 1, "label"])
unassigned_names = set(df_tsv.loc[df_tsv["assigned"] == 0, "label"])

assigned_lengths   = np.array([seq_lengths[n] for n in seq_lengths if n in assigned_names])
unassigned_lengths = np.array([seq_lengths[n] for n in seq_lengths if n in unassigned_names])

# ── Plot ──────────────────────────────────────────────────────────────────────
bins = np.arange(0, 462, 1)

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(assigned_lengths, bins=bins, color="#4E79A7", alpha=0.85,
        label=f"assigned (n={len(assigned_lengths)})")
ax.hist(unassigned_lengths, bins=bins, color="#F28E2B", alpha=0.85,
        label=f"unassigned (n={len(unassigned_lengths)})")

ax.set_yscale("log")
ax.set_xlabel("Sequence length (aa)")
ax.set_ylabel("Count (log scale)")
ax.set_title("agrC sequence length distribution — cluster assignment")
ax.set_xlim(0, 460)
ax.legend(frameon=True, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("agrc_length_hist.png", dpi=300)
print("Saved agrc_length_hist.png")
