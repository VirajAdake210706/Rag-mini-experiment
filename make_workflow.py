"""Workflow diagram v2 for the RAG blog -> fig_workflow.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Polygon, Circle

NAVY, BLUE, TEAL, ORANGE, GREEN, GREY, RED = "#1F3A5F", "#2E5E9E", "#2A9D8F", "#E9A03B", "#2D6A4F", "#6B7280", "#B5483F"
LANE1, LANE2, LANE3 = "#EEF3F9", "#EAF6F4", "#FFF7EA"
EDGE1, EDGE2, EDGE3 = "#C9D6E8", "#BFE3DD", "#F3D9AE"

fig, ax = plt.subplots(figsize=(13.5, 8.4))
ax.set_xlim(0, 14.4); ax.set_ylim(-0.1, 8.6); ax.axis("off"); ax.set_aspect("equal")


def lane(y0, y1, fc, ec, num, title, sub, tcol):
    ax.add_patch(FancyBboxPatch((.2, y0), 14.0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.4))
    ax.add_patch(Circle((.62, y1 - .38), .2, fc=tcol, ec=tcol))
    ax.text(.62, y1 - .38, num, color="white", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(.95, y1 - .38, title, color=tcol, va="center", fontsize=12.5, fontweight="bold")
    ax.text(14.0, y1 - .38, sub.strip(), color=GREY, va="center", ha="right", fontsize=10.5, style="italic")


def box(x, y, w, h, title, sub=None, fc=BLUE, tested=False, icon=None, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=fc, lw=1.5, zorder=3))
    tx = x + w / 2 + (.38 if icon else 0)
    if sub:
        ax.text(tx, y + h / 2 + .17, title, color="white", ha="center", va="center", fontsize=fs, fontweight="bold", zorder=5)
        ax.text(tx, y + h / 2 - .22, sub, color="white", ha="center", va="center", fontsize=8.8, alpha=.92, zorder=5, linespacing=1.25)
    else:
        ax.text(tx, y + h / 2, title, color="white", ha="center", va="center", fontsize=fs, fontweight="bold", zorder=5, linespacing=1.3)
    if icon:
        icon(x + .48, y + h / 2)
    if tested:
        ax.add_patch(FancyBboxPatch((x + w - .82, y + h - .13), .8, .26, boxstyle="round,pad=0.01,rounding_size=0.1", fc=ORANGE, ec="white", lw=1.2, zorder=6))
        ax.text(x + w - .42, y + h, "TESTED", color="white", fontsize=7, fontweight="bold", ha="center", va="center", zorder=7)


def arrow(p, q, color=GREY, lw=2.0, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


# ---- icons (white line art, drawn at box-left)
W = "white"
def i_docs(cx, cy):
    for dx, dy in [(-.07, -.05), (.07, .07)]:
        ax.add_patch(Rectangle((cx - .17 + dx, cy - .22 + dy), .34, .44, fc=W, ec=NAVY, lw=1, zorder=6))
    for k in range(3): ax.plot([cx - .06, cx + .12], [cy + .17 - k * .1] * 2, color="#9FB3D1", lw=1.4, zorder=7)
def i_clean(cx, cy):
    for k, wd in enumerate([.36, .28, .36]): ax.plot([cx - .18, cx - .18 + wd], [cy + .14 - k * .14] * 2, color=W, lw=3, solid_capstyle="round", zorder=6)
def i_chunks(cx, cy):
    for k in range(3): ax.add_patch(FancyBboxPatch((cx - .19, cy + .13 - k * .15), .38, .1, boxstyle="round,pad=0.005,rounding_size=0.03", fc=W, ec=W, zorder=6))
def i_db(cx, cy):
    ax.add_patch(Rectangle((cx - .2, cy - .16), .4, .32, fc=W, ec=W, zorder=6))
    ax.add_patch(Ellipse((cx, cy - .16), .4, .14, fc=W, ec=W, zorder=6)); ax.add_patch(Ellipse((cx, cy + .16), .4, .14, fc="#CFEDE8", ec=W, zorder=7))
    ax.plot([cx - .2, cx + .2], [cy, cy], color=TEAL, lw=1, zorder=8)
def i_q(cx, cy):
    ax.add_patch(FancyBboxPatch((cx - .2, cy - .16), .4, .32, boxstyle="round,pad=0.01,rounding_size=0.08", fc=W, ec=W, zorder=6))
    ax.text(cx, cy - .01, "?", color=NAVY, ha="center", va="center", fontsize=13, fontweight="bold", zorder=7)
def i_kw(cx, cy):
    ax.text(cx, cy, "Aa", color=W, ha="center", va="center", fontsize=14, fontweight="bold", zorder=6)
def i_vec(cx, cy):
    rng = np.random.default_rng(2)
    for x, y in [(-.15, -.1), (-.05, .12), (.1, .05), (.16, -.12), (0, -.02)]: ax.add_patch(Circle((cx + x, cy + y), .045, fc=W, ec=W, zorder=6))
    ax.plot([cx - .15, cx - .05, cx + .1], [cy - .1, cy + .12, cy + .05], color=W, lw=.8, alpha=.7, zorder=6)
def i_rrf(cx, cy):
    ax.text(cx, cy, "Σ", color=W, ha="center", va="center", fontsize=18, fontweight="bold", zorder=6)
def i_funnel(cx, cy):
    ax.add_patch(Polygon([(cx - .22, cy + .18), (cx + .22, cy + .18), (cx + .04, cy - .02), (cx + .04, cy - .2), (cx - .04, cy - .14), (cx - .04, cy - .02)], fc=W, ec=W, zorder=6))
def i_topk(cx, cy):
    for k in range(3): ax.add_patch(FancyBboxPatch((cx - .2, cy + .14 - k * .15), .4, .1, boxstyle="round,pad=0.005,rounding_size=0.03", fc=ORANGE if k == 0 else W, ec=W, zorder=6))
def i_prompt(cx, cy):
    ax.add_patch(Rectangle((cx - .2, cy - .18), .4, .36, fc=W, ec=W, zorder=6))
    for k in range(3): ax.plot([cx - .12, cx + .12], [cy + .09 - k * .09] * 2, color=TEAL, lw=1.6, zorder=7)
def i_llm(cx, cy):
    ax.add_patch(FancyBboxPatch((cx - .15, cy - .15), .3, .3, boxstyle="round,pad=0.01,rounding_size=0.04", fc=W, ec=W, zorder=6))
    for t in (-.08, 0, .08):
        ax.plot([cx + t, cx + t], [cy + .15, cy + .23], color=W, lw=1.6, zorder=6); ax.plot([cx + t, cx + t], [cy - .15, cy - .23], color=W, lw=1.6, zorder=6)
        ax.plot([cx - .15, cx - .23], [cy + t, cy + t], color=W, lw=1.6, zorder=6); ax.plot([cx + .15, cx + .23], [cy + t, cy + t], color=W, lw=1.6, zorder=6)
    ax.text(cx, cy, "AI", color=ORANGE, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=7)
def i_check(cx, cy):
    ax.add_patch(Circle((cx, cy), .2, fc=W, ec=W, zorder=6)); ax.plot([cx - .09, cx - .02, cx + .11], [cy, cy - .08, cy + .09], color=GREEN, lw=2.6, zorder=7, solid_capstyle="round")
def i_x(cx, cy):
    ax.add_patch(Circle((cx, cy), .17, fc=W, ec=W, zorder=6)); ax.text(cx, cy - .01, "?", color=RED, ha="center", va="center", fontsize=12, fontweight="bold", zorder=7)

# ============ Lane 1: prepare knowledge
lane(6.45, 8.55, LANE1, EDGE1, "1", "PREPARE KNOWLEDGE", " offline, run once and refresh when documents change", NAVY)
x1 = [.5, 3.3, 6.1, 8.9, 11.7]
box(x1[0], 6.7, 2.4, 1.2, "Documents", "PDFs, wiki, policies", NAVY, icon=i_docs)
box(x1[1], 6.7, 2.4, 1.2, "Clean & parse", "text + metadata", BLUE, icon=i_clean)
box(x1[2], 6.7, 2.4, 1.2, "Chunk", "split + add title,\noverlap", BLUE, tested=True, icon=i_chunks)
box(x1[3], 6.7, 2.4, 1.2, "Embed", "text → vectors", TEAL, icon=i_vec)
box(x1[4], 6.7, 2.4, 1.2, "Search index", "keyword + vector", TEAL, icon=i_db)
for i in range(4): arrow((x1[i] + 2.43, 7.3), (x1[i + 1] - .03, 7.3))

# ============ Lane 2: find evidence
lane(3.35, 6.2, LANE2, EDGE2, "2", "FIND EVIDENCE", " online, for every question", TEAL)
box(.5, 3.95, 2.0, 1.3, "User\nquestion", None, NAVY, icon=i_q)
box(3.3, 4.65, 2.4, .95, "Keyword", "BM25 exact terms", BLUE, tested=True, icon=i_kw, fs=10.5)
box(3.3, 3.5, 2.4, .95, "Vector", "meaning match", BLUE, tested=True, icon=i_vec, fs=10.5)
box(6.5, 3.95, 2.0, 1.3, "Fuse", "RRF merges\nby rank", TEAL, tested=True, icon=i_rrf)
box(9.2, 3.95, 2.2, 1.3, "Rerank", "top 20 → top k", TEAL, tested=True, icon=i_funnel)
box(12.0, 3.95, 2.0, 1.3, "Top-k\nchunks", None, GREEN, icon=i_topk)
arrow((2.53, 4.9), (3.27, 5.12), rad=-0.1); arrow((2.53, 4.3), (3.27, 3.98), rad=0.1)
arrow((5.73, 5.12), (6.47, 4.85), rad=-0.1); arrow((5.73, 3.98), (6.47, 4.35), rad=0.1)
arrow((8.53, 4.6), (9.17, 4.6)); arrow((11.43, 4.6), (11.97, 4.6))
# index feeds retrieval
ax.plot([12.9, 12.9, 4.5], [6.7, 6.33, 6.33], color=TEAL, lw=1.7, zorder=1)
arrow((4.5, 6.33), (4.5, 5.64), color=TEAL, lw=1.7)
ax.text(7.6, 6.04, "both searches query this index", color=TEAL, fontsize=9.5, ha="center", style="italic", zorder=5)

# ============ Lane 3: answer with proof
lane(.7, 3.15, LANE3, EDGE3, "3", "ANSWER WITH PROOF", " online, grounded in the retrieved text", "#9A5B00")
ax.add_patch(FancyBboxPatch((.5, .95), 2.8, 1.45, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=EDGE3, lw=1.3, zorder=3))
ax.text(1.9, 2.12, "Guard rails in the prompt", ha="center", color="#9A5B00", fontsize=9.5, fontweight="bold", zorder=5)
ax.text(1.9, 1.48, "• answer only from sources\n• cite chunk ids\n• say \"Not found\" if unsure\n• respect user permissions", ha="center", va="center", color="#6b4a12", fontsize=8.6, linespacing=1.35, zorder=5)
box(3.9, 1.2, 2.6, 1.2, "Build prompt", "rules + chunks\n+ question", TEAL, icon=i_prompt)
box(7.2, 1.2, 2.0, 1.2, "LLM", "writes answer", ORANGE, icon=i_llm)
box(10.1, 1.68, 3.9, .82, "Answer + citations [1][2]", "when the evidence supports it", GREEN, icon=i_check, fs=10.5)
box(10.1, .82, 3.9, .75, "\"Not found\"", "when evidence is missing", "#8C8F99", icon=i_x, fs=10.5)
arrow((6.53, 1.8), (7.17, 1.8)); arrow((9.23, 2.0), (10.07, 2.04), rad=-0.1); arrow((9.23, 1.6), (10.07, 1.2), rad=0.1)
# chunks go into prompt
ax.plot([13.0, 13.0, 5.2], [3.95, 3.25, 3.25], color=GREEN, lw=1.7, zorder=1)
arrow((5.2, 3.25), (5.2, 2.44), color=GREEN, lw=1.7)

# ============ evaluation strip + legend
ax.add_patch(FancyBboxPatch((.2, -.05), 14.0, .5, boxstyle="round,pad=0.01,rounding_size=0.12", fc="#F3F4F6", ec="#D9DCE1", lw=1.2))
ax.text(.45, .2, "MEASURE EACH STAGE", color=NAVY, fontsize=9.5, fontweight="bold", va="center")
ax.text(3.0, .2, "Retrieval: Recall@k, MRR (this post)   ·   Answers: faithfulness, citation accuracy, abstention (next step)", color=GREY, fontsize=9.5, va="center")
ax.add_patch(FancyBboxPatch((12.05, .09), .7, .22, boxstyle="round,pad=0.01,rounding_size=0.1", fc=ORANGE, ec=ORANGE))
ax.text(12.4, .2, "TESTED", color="white", fontsize=7, fontweight="bold", ha="center", va="center")
ax.text(12.85, .2, "= in experiment", color=GREY, fontsize=8.5, va="center")

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig_workflow.png", dpi=170, bbox_inches="tight", pad_inches=.15, facecolor="white")
