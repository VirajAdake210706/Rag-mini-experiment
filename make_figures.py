"""Hero banner and result charts for the blog, drawn from results/results.json (run rag_experiment.py first)."""
import json, os
os.makedirs("figures", exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

R = json.load(open("results/results.json"))["by_chunk"]
NAVY, BLUE, TEAL, ORANGE, RED, GREY, LIGHT = "#1F3A5F", "#2E5E9E", "#2A9D8F", "#E9A03B", "#C44E52", "#6B7280", "#EEF3F9"
M = ["Keyword (BM25)", "Dense (LSA)", "Hybrid (RRF)", "Hybrid + rerank"]
MC = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=11, bold=True, r=0.08):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec or fc, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=GREY, rad=0.0, lw=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ hero banner
fig, ax = plt.subplots(figsize=(12, 6.3))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
fig.patch.set_facecolor(NAVY)
ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color=NAVY))
rng = np.random.default_rng(3)
for _ in range(60):  # faint background dots
    ax.add_patch(Circle((rng.uniform(0, 12), rng.uniform(0, 6.3)), rng.uniform(.02, .06), color="#2B4C7A", alpha=.8))
# document stack
for i, (dx, dy) in enumerate([(0, 0), (.18, .18), (.36, .36)]):
    ax.add_patch(FancyBboxPatch((.9 + dx, 1.9 + dy), 1.7, 2.2, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="white", ec="#C9D6E8", lw=1.2, zorder=2 + i))
for k in range(6):
    ax.plot([1.15 + .36, 2.2 + .36], [3.7 - k * .3 + .36] * 2, color="#B8C7DD", lw=3, solid_capstyle="round", zorder=6)
ax.text(1.95, 1.45, "Your documents", color="#C9D6E8", ha="center", fontsize=12)
# chunks
for i in range(5):
    y = 4.4 - i * .72
    hit = i == 2
    ax.add_patch(FancyBboxPatch((4.1, y), 1.9, .5, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=ORANGE if hit else "#2B4C7A", ec=ORANGE if hit else "#4A6FA5", lw=1.5))
    ax.plot([4.3, 5.6], [y + .25] * 2, color="white" if hit else "#7F9CC8", lw=3, solid_capstyle="round")
ax.text(5.05, 1.1, "Retrieved chunks", color="#C9D6E8", ha="center", fontsize=12)
# answer card
ax.add_patch(FancyBboxPatch((8.0, 2.0), 3.0, 2.3, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec="white"))
ax.text(8.25, 3.85, "Answer", color=NAVY, fontsize=13, fontweight="bold")
for k, wd in enumerate([2.3, 2.0, 1.4]):
    ax.plot([8.25, 8.25 + wd], [3.45 - k * .3] * 2, color="#B8C7DD", lw=4, solid_capstyle="round")
ax.add_patch(FancyBboxPatch((8.25, 2.25), .75, .35, boxstyle="round,pad=0.01,rounding_size=0.1", fc=ORANGE, ec=ORANGE))
ax.text(8.625, 2.425, "[1] source", color="white", fontsize=9, fontweight="bold", ha="center", va="center")
ax.text(9.5, 1.45, "Cited & checkable", color="#C9D6E8", ha="center", fontsize=12)
arrow(ax, 3.2, 3.0, 3.95, 3.0, color="#C9D6E8", lw=3)
arrow(ax, 6.2, 3.0, 7.85, 3.0, color="#C9D6E8", lw=3)
ax.text(3.58, 3.25, "search", color="#C9D6E8", ha="center", fontsize=10)
ax.text(7.0, 3.25, "LLM + evidence", color="#C9D6E8", ha="center", fontsize=10)
ax.text(.6, 5.65, "Your Enterprise Chatbot Is Guessing", color="white", fontsize=25, fontweight="bold")
ax.text(.6, 5.15, "Retrieval-Augmented Generation, tested on 120 questions", color="#C9D6E8", fontsize=14)
plt.savefig("figures/hero_banner.png", dpi=150, facecolor=NAVY, bbox_inches="tight", pad_inches=0.1); plt.close()

# ------------------------------------------------------------------ heatmap recall@3
cfg = ["25w", "50w", "100w", "200w", "50w+title"]
lab = ["25 words", "50 words", "100 words", "200 words", "50 words\n+ title & overlap"]
Z = np.array([[R[c][m]["r3"] for c in cfg] for m in M])
fig, ax = plt.subplots(figsize=(9, 4.2))
im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(cfg))); ax.set_xticklabels(lab, fontsize=10); ax.set_yticks(range(4)); ax.set_yticklabels(M)
for i in range(4):
    for j in range(len(cfg)):
        ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="white" if Z[i, j] > .55 else NAVY, fontsize=12, fontweight="bold")
ax.spines[:].set_visible(False); ax.tick_params(length=0)
ax.set_title("Recall@3 for every retriever × chunking setup", pad=12)
fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="Recall@3")
plt.tight_layout(); plt.savefig("figures/fig3_heatmap_recall3.png", dpi=160); plt.close()

# ------------------------------------------------------------------ recall@k at 100w
fig, ax = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(3); w = .2
for i, (m, c) in enumerate(zip(M, MC)):
    vals = [R["100w"][m][k] for k in ("r1", "r3", "r5")]
    ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=m)
    for xi, v in zip(x + (i - 1.5) * w, vals):
        ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(["Recall@1", "Recall@3", "Recall@5"]); ax.set_ylim(0, 1.0)
ax.set_title("100-word chunks: how often is the right evidence in the top k?")
ax.legend(frameon=False, ncol=2, loc="upper left"); ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig("figures/fig4_recall_at_k_100w.png", dpi=160); plt.close()

# ------------------------------------------------------------------ cost vs quality
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for m, c in zip(M, MC):
    ax.plot([R[k]["n_chunks"] for k in cfg[:4]], [R[k][m]["mrr"] for k in cfg[:4]], marker="o", color=c, lw=2, label=m)
ax.scatter([R["50w+title"]["n_chunks"]], [R["50w+title"]["Hybrid + rerank"]["mrr"]], s=170, marker="*", color=NAVY, zorder=5)
ax.annotate("50-word chunks + title & overlap\n(cheap fix, hybrid + rerank)", (R["50w+title"]["n_chunks"], R["50w+title"]["Hybrid + rerank"]["mrr"]),
            xytext=(145, .7), arrowprops=dict(arrowstyle="->", color=NAVY), fontsize=9.5, color=NAVY)
for k, lbl in zip(cfg[:4], ["25w", "50w", "100w", "200w"]):
    ax.annotate(lbl, (R[k]["n_chunks"], R[k]["Hybrid + rerank"]["mrr"]), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=GREY)
ax.set_xlabel("Number of chunks in the index (more chunks = larger index, more candidates to sift)")
ax.set_ylabel("MRR (higher = right chunk ranked earlier)")
ax.set_title("Index size vs. ranking quality"); ax.legend(frameon=False, loc="lower left"); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig("figures/fig5_index_size_vs_mrr.png", dpi=160); plt.close()
print("done")
