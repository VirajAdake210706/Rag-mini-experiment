# RAG Mini-Experiment: Does Chunking and Retrieval Strategy Matter?

Code, data generator and figures for the blog post **"Your Enterprise Chatbot Is Guessing: A Hands-On Look at Retrieval-Augmented Generation"**.

Blog post: [https://medium.com/@adakeviraj2107/your-enterprise-chatbot-is-guessing-a-hands-on-look-at-retrieval-augmented-generation-b35eae1a5033?postPublishedType=repub]

## What this repo does

It measures the **retrieval stage** of a Retrieval-Augmented Generation (RAG) pipeline on a fully **synthetic enterprise-policy corpus**, and compares:

- **chunk sizes** (25, 50, 100 and 200 words) and a "context-restored" setup (50-word chunks + document title + 10-word overlap)
- **four retrievers**: keyword (BM25), "dense" (TF-IDF + LSA), hybrid (Reciprocal Rank Fusion) and hybrid + rerank

| | |
|---|---|
| Corpus | 40 synthetic policy documents (8 policy types x 5 regions), about 118 words each |
| Questions | 120 paraphrased questions, one per numeric fact |
| Hit definition | the retrieved chunk is from the correct document **and** contains the answer sentence |
| Metrics | Recall@1, @3, @5 and MRR |
| Determinism | fixed seed (7); results reproduce exactly |
| Runtime | under 10 seconds on a laptop, no GPU, no internet, no API keys |

## Headline results (Recall@3)

| Retriever | 25 words | 50 words | 100 words | 200 words | 50 words + title & overlap |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.31 | 0.44 | 0.78 | 0.96 | 0.69 |
| Dense (LSA) | 0.25 | 0.42 | 0.68 | 0.92 | 0.70 |
| Hybrid (RRF) | 0.28 | 0.44 | 0.78 | 0.95 | 0.75 |
| Hybrid + rerank | 0.28 | 0.47 | 0.81 | 0.96 | 0.82 |

Full numbers (Recall@1/3/5, MRR, chunk counts) are in `results/results.json`.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── rag_experiment.py      # corpus generator, retrievers, evaluation -> results/results.json
├── make_figures.py        # hero banner + result charts (reads results/results.json)
├── make_workflow.py       # RAG workflow diagram
├── trace_example.py       # follows one question through every pipeline stage
├── results/
│   ├── results.json       # all metrics
│   └── trace_example_output.txt
├── figures/               # all images used in the blog
└── blog/
    └── rag_blog.md        # full blog text (markdown)
```

## How to reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python rag_experiment.py           # runs the experiment, writes results/results.json and 2 charts
python make_figures.py             # writes the hero banner and the other charts
python make_workflow.py            # writes the workflow diagram
python trace_example.py            # prints a stage-by-stage trace of one question
```

Run the commands from the repository root. `make_figures.py` needs `results/results.json`, so run `rag_experiment.py` first.

## Important limitations

- **Synthetic data.** Real documents are messier (tables, scans, conflicting versions). Expect lower absolute scores.
- **Stand-ins, not production components.** The "dense" retriever uses LSA instead of a neural embedding model, and the reranker scores query-term coverage instead of using a cross-encoder. They keep the experiment fully offline; ranking of methods could change with real components.
- **Retrieval only.** No language model is called, so this repo makes no claim about how much RAG reduces hallucination. The blog post describes a protocol for the generation-side comparison (with vs. without retrieval, scored for exact match, faithfulness and abstention).
- **Small scale.** 40 documents and 120 questions reveal large effects (such as chunking), not differences of a few points.
- The 200-word setting is effectively whole-document retrieval because documents average about 118 words.

## Ideas for extending it

- Replace LSA with a real embedding model (for example, a sentence-transformers model) and the lexical reranker with a cross-encoder.
- Add metadata filtering on region and measure the effect on near-duplicate documents.
- Add the generation step with an LLM and score faithfulness, citation accuracy and abstention.
- Try structure-aware chunking (by heading or section) on real documents.

## References

1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. https://arxiv.org/abs/2005.11401
2. Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4). https://doi.org/10.1561/1500000019
3. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP. https://arxiv.org/abs/2004.04906
4. Cormack, G. V., Clarke, C. L. A. and Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR. https://doi.org/10.1145/1571941.1572114
5. Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. https://arxiv.org/abs/1901.04085
6. Es, S. et al. (2024). RAGAs: Automated Evaluation of Retrieval Augmented Generation. EACL (System Demonstrations). https://arxiv.org/abs/2309.15217
7. Anthropic (2024). Introducing Contextual Retrieval. https://www.anthropic.com/news/contextual-retrieval
8. Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL, 12. https://doi.org/10.1162/tacl_a_00638

## License

MIT. See `LICENSE`. All data in this repository is synthetic and generated by the code.
