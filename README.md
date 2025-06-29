# semantic_tags
a method for parsing conversations and using clustering to determine the number of tags needed then linking threads from multiple pieces of text together for retrieval and sorting



Semantic Tagging & Knowledge Graph System

1 – Purpose

Create a modular pipeline that automatically extracts, summarises, and semantically tags conversation fragments, builds an evolving tag graph, and surfaces insights through an interactive UI. The system should work offline (Float‑friendly), be extensible, and remain model‑agnostic.

2 – High‑Level Pipeline

Ingestion – Pull raw transcripts (Markdown, JSON, etc.).

Chunking – Split into semantically coherent "nuggets" (rule‑based + language‑model fallback).

Vectorisation – Embed each nugget (e.g. sentence-transformers/all-MiniLM-L6-v2 or local OpenAI compatible model).

Initial Tagging

Heuristic labels: regex/keyword dictionaries for obvious domains ("recipe", "anime", etc.).

k‑NN label propagation: nearest neighbours inherit tags.

Clustering – Unsupervised grouping to surface emergent themes.

Graph Construction – Nodes = tags + nuggets; edges = co‑occurrence / similarity.

Summarisation – LLM generates tag‑level • day/ week/ month digests.

UI Layer – Interactive graph + timeline + tag browser.

3 – Data Model

graph LR
  subgraph VectorStore (Weaviate / Qdrant)
  Nugget-->|embedding|Vector
  end
  Nugget:::msg -->|has_tag| Tag
  Tag -. co_occurs .-> Tag

Nugget {id, text, ts, embed, tags[], cluster_id}

Tag   {id, name, first_seen, last_seen, count}

Cluster {id, centroid, score, label_hint}

4 – Chunking Details

Hard breaks: user/assistant message boundaries.

Soft breaks: sentence splits when token ≥ 128 OR topic shift detected (cosine ≤ 0.75 with running mean).

Merge too‑small (< 20 tokens) chunks with neighbours.

5 – Clustering & Optimal k

Option

Library

k‑selection

K‑Means

scikit‑learn

Elbow + Silhouette (choose first local max > 0.25)

HDBSCAN

hdbscan

Density‑based; let min_cluster_size ≈ √N

Community‑detection on k‑NN graph (Leiden)

igraph/leidenalg

Modularity maximisation – no k needed

Pseudo‑formula (Elbow):

For k = 2…√N:
  WCSS_k = Σ dist(point, centroid_k)^2
Choose k where Δ(WCSS_{k‑1}‑WCSS_k) < τ

Typical τ = 5 % of previous Δ.

6 – Tag Imputation

For each cluster, aggregate top‑N TF‑IDF tokens.

Run zero‑shot classifier (e.g. facebook/bart‑mnli) against user seed tag list.

If max score < σ (e.g. 0.6) ⇒ propose new tag candidate.

7 – Graph & Timeline UI

Stack: React + D3, Tailwind for styling.

Views:

Galaxy Graph – draggable force layout; node size ~ mention count; coloured by manual vs auto tag.

Timeline Heatmap – x=time, y=tags, colour=intensity.

Diff View – pick two periods → show new/vanishing tags & summary deltas.

8 – Consistency Review Workflow

Daily cron summarises yesterday per tag.

Weekly task highlights tags with ≥ 20 % sentiment shift or cluster drift (centroid moved > ε).

UI badge prompts user to accept / merge / rename / delete anomalous tags.

9 – APIs & Storage Choices

Vector store: Weaviate (float‑compatible, offers HNSW + hybrid search).

Graph DB: Neo4j optional; otherwise derive edges on the fly.

Orchestration: FastAPI backend; background workers with Celery; expose /ingest, /query, /summary.

10 – Future Extensions

Speaker diarisation & emotion embeddings.

RAG loop: ask LLM "Given current tag graph, what did I forget to tag?"

Fine‑tune small classifier on accepted tags for faster online labelling.

Last updated: 2025‑06‑16


## Usage

Install dependencies via `pip install -r requirements.txt`. Then run:

```bash
python -m semantic_tags.cli path/to/transcripts --tags=tag1,tag2 --summary-out summary.json
```

The CLI now supports custom tag lists, embedding batch size and device options, and saving summaries or uploading to a Weaviate instance via `--weaviate-url`.

## Tests

Run the test suite with:

```bash
pytest
```
