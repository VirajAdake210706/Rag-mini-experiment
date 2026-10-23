"""Trace one question through every stage of the experiment pipeline (50-word chunks + title + overlap)."""
import io, contextlib, json
with contextlib.redirect_stdout(io.StringIO()):
    import rag_experiment as R   # re-runs the (deterministic) experiment and exposes its functions

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

chunks = R.build(50, 10, True)
texts = [c["text"] for c in chunks]
Q = next(q for q in R.questions if q["q"].startswith("How many days a week can employees in India work from home"))
print("QUESTION:", Q["q"]); print("GOLD FACT:", Q["fact"]); print("GOLD DOC:", R.docs[Q["doc"]]["text"][:60], "\n")

bm = R.BM25(texts)
b = bm.scores(Q["q"]); bm_rank = list(np.argsort(-b))
tf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2)).fit(texts + [q["q"] for q in R.questions])
X = tf.transform(texts); svd = TruncatedSVD(n_components=128, random_state=7).fit(X)
E = svd.transform(X); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
qe = svd.transform(tf.transform([Q["q"]]))[0]; qe /= (np.linalg.norm(qe) + 1e-9)
sims = E @ qe; d_rank = list(np.argsort(-sims))
h_rank = R.rrf([bm_rank[:50], d_rank[:50]])
qt = set(R.tok(Q["q"])) - {"what", "is", "the", "in", "how", "many", "of", "a", "for", "does", "do", "are", "to", "be", "must", "at", "after", "within", "per"}
rr = lambda i: len(qt & set(R.tok(texts[i]))) / len(qt)
rer = sorted(h_rank[:20], key=lambda i: -rr(i)) + h_rank[20:]

def show(name, rank, score=None):
    print("==", name)
    for pos, i in enumerate(rank[:3], 1):
        hit = R.is_hit(chunks[i], Q)
        sc = f"  score={score(i):.3f}" if score else ""
        print(f" #{pos} chunk{i} {'<-- HIT' if hit else ''}{sc}\n    {texts[i][:170]}")
    first = next((p for p, i in enumerate(rank[:50], 1) if R.is_hit(chunks[i], Q)), None)
    print("   first hit rank:", first, "\n")

print("chunks in index:", len(chunks))
show("KEYWORD (BM25)", bm_rank, lambda i: b[i])
show("VECTOR (LSA)", d_rank, lambda i: sims[i])
rrfs = {i: sum(1 / (60 + r + 1) for rl in (bm_rank[:50], d_rank[:50]) for r, j in enumerate(rl) if j == i) for i in h_rank[:5]}
show("FUSED (RRF)", h_rank, lambda i: rrfs.get(i, 0))
show("RERANKED", rer, rr)
gold = next(i for i in range(len(chunks)) if R.is_hit(chunks[i], Q))
print("gold chunk", gold, "ranks -> bm25:", bm_rank.index(gold) + 1, "vec:", d_rank.index(gold) + 1, "fused:", h_rank.index(gold) + 1, "rerank:", rer.index(gold) + 1)
print("RRF for gold:", round(1 / (60 + bm_rank.index(gold) + 1) + 1 / (60 + d_rank.index(gold) + 1), 4))
