"""Mini-experiment: how much do chunking and retrieval strategy matter for RAG?
Fully synthetic corpus, fully offline, deterministic (seed 7)."""
import random, re, json, math, os
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(7); np.random.seed(7)

REGIONS = ["APAC", "EMEA", "North America", "Latin America", "India"]
# topic -> (boilerplate sentences, [(doc phrasing, question phrasing, value fn)])
TOPICS = {
 "Travel": ([
   "All travel must be booked through the approved corporate portal unless an exception is granted.",
   "Employees are expected to choose the most economical fare that meets business needs.",
   "Receipts must be submitted within the reporting window to be eligible for reimbursement.",
   "Managers review travel requests for business justification before approval.",
   "Personal travel combined with business travel must be clearly separated on the claim."],
  [("The per diem food allowance is {v} USD per day.", "What is the daily meal limit for travel in {r}?", lambda: random.randint(40, 95)),
   ("Hotel stays are capped at {v} USD per night.", "What is the maximum nightly hotel rate in {r}?", lambda: random.randint(120, 320)),
   ("Flights longer than {v} hours qualify for premium economy.", "After how many flight hours can staff in {r} fly premium economy?", lambda: random.randint(5, 9))]),
 "Remote Work": ([
   "Remote work arrangements must be agreed with the line manager and recorded in the HR system.",
   "Employees must work from a location with a secure and reliable internet connection.",
   "Company equipment must be used for all work involving confidential data.",
   "Team meetings should remain accessible to remote participants.",
   "Performance expectations are identical for remote and office-based staff."],
  [("Staff may work remotely up to {v} days per week.", "How many days a week can employees in {r} work from home?", lambda: random.randint(1, 4)),
   ("The home office stipend is {v} USD per year.", "What yearly budget is given for a home workspace in {r}?", lambda: random.randint(200, 900)),
   ("Core collaboration hours are {v} hours per day.", "How many overlapping hours per day must remote staff in {r} be available?", lambda: random.randint(3, 6))]),
 "Data Retention": ([
   "Records must be stored in approved systems and never on personal devices.",
   "Legal hold notices override all standard deletion schedules.",
   "Data owners are accountable for classifying records at creation.",
   "Deletion must be performed using approved secure erasure procedures.",
   "Retention schedules are reviewed annually by the compliance office."],
  [("Customer records are kept for {v} years after account closure.", "How long are client files retained after an account ends in {r}?", lambda: random.randint(3, 10)),
   ("System access logs are retained for {v} months.", "For how many months are login audit trails kept in {r}?", lambda: random.randint(6, 36)),
   ("Email archives are retained for {v} years.", "How many years are mailbox backups preserved in {r}?", lambda: random.randint(2, 8))]),
 "Procurement": ([
   "Purchases must follow the approved vendor list wherever a listed supplier exists.",
   "Conflicts of interest must be disclosed before any evaluation begins.",
   "Contracts are reviewed by legal prior to signature.",
   "Purchase orders must be raised before goods or services are received.",
   "Splitting a purchase to avoid approval thresholds is prohibited."],
  [("Purchases above {v} thousand USD require three competitive quotes.", "Above what purchase value do buyers in {r} need three bids?", lambda: random.randint(5, 50)),
   ("Director approval is required above {v} thousand USD.", "At what spend level must a director sign off in {r}?", lambda: random.randint(25, 200)),
   ("Standard supplier payment terms are {v} days.", "How many days do suppliers in {r} get to be paid?", lambda: random.randint(30, 75))]),
 "Security Incident": ([
   "Any suspected incident must be reported to the security operations centre immediately.",
   "Affected systems must not be powered off until forensic guidance is received.",
   "Communication about an incident is restricted to the response team.",
   "Post-incident reviews are mandatory for all severity one and two events.",
   "Evidence must be preserved following chain of custody rules."],
  [("Severity one incidents must be acknowledged within {v} minutes.", "Within how many minutes must a critical breach be acknowledged in {r}?", lambda: random.randint(10, 60)),
   ("Regulators must be notified within {v} hours of confirmation.", "How many hours do teams in {r} have to inform the regulator?", lambda: random.randint(24, 96)),
   ("Post-incident reports are due within {v} business days.", "How many working days after an event is the review report due in {r}?", lambda: random.randint(3, 15))]),
 "Leave": ([
   "Leave requests should be submitted through the HR portal in advance.",
   "Approval depends on team coverage and business needs.",
   "Public holidays are not counted against leave balances.",
   "Unused leave handling follows local labour regulation.",
   "Employees on leave are not expected to monitor work messages."],
  [("Full-time staff receive {v} days of annual leave.", "How many vacation days do full-time employees get in {r}?", lambda: random.randint(18, 30)),
   ("Parental leave lasts {v} weeks.", "What is the length of leave for new parents in {r}?", lambda: random.randint(8, 26)),
   ("Up to {v} days of leave may be carried over.", "How many unused days can be rolled into next year in {r}?", lambda: random.randint(3, 12))]),
 "Training": ([
   "Learning plans are agreed with managers during the annual planning cycle.",
   "Mandatory courses must be completed before their deadline.",
   "External courses require a business case and manager approval.",
   "Certification costs are reimbursed after successful completion.",
   "Training records are maintained in the learning platform."],
  [("Each employee has a learning budget of {v} USD per year.", "What annual education allowance does each person in {r} have?", lambda: random.randint(500, 2500)),
   ("Employees may take {v} paid study days per year.", "How many paid days for studying are allowed in {r}?", lambda: random.randint(2, 8)),
   ("Compliance courses must be renewed every {v} months.", "How often, in months, must mandatory courses be repeated in {r}?", lambda: random.randint(6, 24))]),
 "Vendor Onboarding": ([
   "New vendors must complete due diligence before being added to the system.",
   "Banking details are verified through an independent callback.",
   "Vendors must accept the supplier code of conduct.",
   "Sanctions screening is performed on every new supplier.",
   "Onboarding records are reviewed by the finance team."],
  [("Vendor due diligence must be completed within {v} business days.", "How many working days does supplier vetting take in {r}?", lambda: random.randint(5, 20)),
   ("Vendor risk reassessment occurs every {v} months.", "How frequently, in months, are suppliers re-evaluated in {r}?", lambda: random.randint(6, 24)),
   ("Insurance coverage of at least {v} million USD is required.", "What minimum insurance cover must a supplier in {r} hold?", lambda: random.randint(1, 10))]),
}

FILLER = ["This policy applies to all permanent and contract employees.",
 "Exceptions must be documented and approved in writing.",
 "Questions about this policy should be directed to the relevant function.",
 "The policy owner reviews this document at least once a year.",
 "Employees who violate this policy may face disciplinary action.",
 "Local law prevails where it conflicts with this policy.",
 "Training on this policy is provided during onboarding.",
 "Updates to this policy are announced through internal channels."]

docs, questions = [], []
for topic, (boiler, facts) in TOPICS.items():
    for r in REGIONS:
        sents = []
        picked = []
        for dp, qp, fn in facts:
            v = fn()
            picked.append((dp.format(v=v), qp.format(r=r), str(v)))
        body = [f"{topic} Policy for {r}."] + random.sample(boiler, 5) + random.sample(FILLER, 4)
        # facts placed at random positions in the document, region named only once at the top
        for fs, _, _ in picked:
            body.insert(random.randint(1, len(body)), fs)
        # make facts carry the region only in the heading: mimic real policies
        did = len(docs)
        docs.append({"id": did, "text": " ".join(body)})
        for fs, q, v in picked:
            questions.append({"q": q, "doc": did, "fact": fs})

def chunk(text, size, overlap):
    w = text.split(); out = []; i = 0
    while i < len(w):
        out.append(" ".join(w[i:i+size]))
        if i + size >= len(w): break
        i += size - overlap
    return out

def build(size, overlap=0, prefix_title=False):
    chunks = []
    for d in docs:
        title = d["text"].split(".")[0] + "."
        for c in chunk(d["text"], size, overlap):
            chunks.append({"doc": d["id"], "text": (title + " " + c) if prefix_title and not c.startswith(title) else c})
    return chunks

def is_hit(ch, q):
    return ch["doc"] == q["doc"] and q["fact"] in ch["text"]

tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tok(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.docs]); self.avg = self.len.mean()
        df = Counter(); [df.update(c.keys()) for c in self.docs]
        n = len(texts); self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}
        self.k1, self.b = k1, b
    def scores(self, q):
        s = np.zeros(len(self.docs))
        for t in set(tok(q)):
            if t not in self.idf: continue
            for i, c in enumerate(self.docs):
                f = c.get(t, 0)
                if f: s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

def rrf(rank_lists, k=60):
    sc = Counter()
    for rl in rank_lists:
        for r, i in enumerate(rl): sc[i] += 1 / (k + r + 1)
    return [i for i, _ in sc.most_common()]

def evaluate(chunks, use_title_prefix_note=""):
    texts = [c["text"] for c in chunks]
    bm = BM25(texts)
    tf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2)).fit(texts + [q["q"] for q in questions])
    X = tf.transform(texts)
    svd = TruncatedSVD(n_components=min(128, X.shape[1] - 1), random_state=7).fit(X)
    E = svd.transform(X); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    res = {m: {"r1": 0, "r3": 0, "r5": 0, "mrr": 0} for m in ["Keyword (BM25)", "Dense (LSA)", "Hybrid (RRF)", "Hybrid + rerank"]}
    for q in questions:
        b = bm.scores(q["q"]); bm_rank = list(np.argsort(-b))
        qe = svd.transform(tf.transform([q["q"]]))[0]; qe /= (np.linalg.norm(qe) + 1e-9)
        d_rank = list(np.argsort(-(E @ qe)))
        h_rank = rrf([bm_rank[:50], d_rank[:50]])
        qt = set(tok(q["q"])) - {"what", "is", "the", "in", "how", "many", "of", "a", "for", "does", "do", "are", "to", "be", "must", "at", "after", "within", "per"}
        def rr_score(i):
            ct = set(tok(texts[i])); return len(qt & ct) / (len(qt) + 1e-9)
        rer = sorted(h_rank[:20], key=lambda i: -rr_score(i)) + h_rank[20:]
        for name, rank in zip(res, [bm_rank, d_rank, h_rank, rer]):
            hits = [is_hit(chunks[i], q) for i in rank[:10]]
            for k in (1, 3, 5): res[name][f"r{k}"] += any(hits[:k])
            for j, h in enumerate(hits):
                if h: res[name]["mrr"] += 1 / (j + 1); break
    n = len(questions)
    return {m: {k: round(v / n, 3) for k, v in d.items()} for m, d in res.items()}

out = {"n_docs": len(docs), "n_questions": len(questions),
       "avg_words": round(np.mean([len(d["text"].split()) for d in docs]), 1)}
configs = {"25w": (25, 0, False), "50w": (50, 0, False), "100w": (100, 0, False), "200w": (200, 0, False), "50w+title": (50, 10, True)}
out["by_chunk"] = {}
for name, (s, o, t) in configs.items():
    ch = build(s, o, t)
    out["by_chunk"][name] = {"n_chunks": len(ch), **evaluate(ch)}
    print(name, len(ch), out["by_chunk"][name]["Hybrid + rerank"], flush=True)

json.dump(out, open("results/results.json", "w"), indent=1)

# ---- plots
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
methods = ["Keyword (BM25)", "Dense (LSA)", "Hybrid (RRF)", "Hybrid + rerank"]
sizes = ["25w", "50w", "100w", "200w"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for m, c in zip(methods, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
    ax.plot(["25", "50", "100", "200"], [out["by_chunk"][s][m]["r3"] for s in sizes], marker="o", label=m, color=c, lw=2)
ax.set_xlabel("Chunk size (words)"); ax.set_ylabel("Recall@3"); ax.set_ylim(0, 1.02)
ax.set_title("Chunk size and retriever vs. Recall@3 (120 questions)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig1_chunk_size_recall.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(methods)); w = .38
a = [out["by_chunk"]["50w"][m]["r3"] for m in methods]
b = [out["by_chunk"]["50w+title"][m]["r3"] for m in methods]
ax.bar(x - w/2, a, w, label="Plain 50-word chunks", color="#8c9bb5")
ax.bar(x + w/2, b, w, label="+ document title prepended, 10-word overlap", color="#2E5E9E")
for i, (u, v) in enumerate(zip(a, b)):
    ax.text(i - w/2, u + .01, f"{u:.2f}", ha="center", fontsize=9); ax.text(i + w/2, v + .01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0, 1.1); ax.set_ylabel("Recall@3")
ax.set_title("Adding context to each chunk"); ax.legend(frameon=False, loc="upper left"); plt.tight_layout()
plt.savefig("figures/fig2_context_prefix.png", dpi=160); plt.close()
