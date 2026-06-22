"""Generate presentation figures from the GENUINE full-PALM results (no fabrication)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/image_results_linear_eval.json") as f:
    r = json.load(f)

# ---- Figure: performance metrics bar chart ----
metrics = {"AUC": r["auc"], "Accuracy": r["accuracy"],
           "Sensitivity": r["sensitivity"], "Specificity": r["specificity"]}
fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(list(metrics), [v * 100 for v in metrics.values()],
              color=["#2c6fbb", "#3a9679", "#e08a3c", "#9b59b6"], width=0.6)
ax.set_ylim(0, 115)                       # headroom so labels never touch the title
ax.set_ylabel("Percent (%)")
ax.set_title(f"Image model on FULL PALM test set (n={r['n_test']})", pad=14)
for b, v in zip(bars, metrics.values()):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 2.0, f"{v*100:.1f}",
            ha="center", va="bottom", fontweight="bold")
plt.tight_layout(); plt.savefig("results/metrics_bar.png", dpi=150); plt.close()

# ---- Figure: confusion matrix ----
cm = r["confusion_matrix"]
mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(mat, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nhealthy", "Predicted\nmyopic"])
ax.set_yticklabels(["Truly\nhealthy", "Truly\nmyopic"])
ax.set_title(f"Confusion matrix (test set, n={r['n_test']})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=18,
                fontweight="bold", color="white" if mat[i, j] > mat.max()/2 else "black")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()
print("Saved metrics_bar.png and confusion_matrix.png from full-PALM results")
