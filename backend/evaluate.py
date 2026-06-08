"""
Offline evaluation with the RAGAS framework.

Flow (run from anywhere, like ingest.py):
    1. load the test set (data/eval_set.json: question + ground_truth)
    2. for each config (Semantic / +BM25 / +Reranker), run the pipeline on
       every question to collect the generated answer + retrieved contexts
    3. score each config with RAGAS on 4 metrics:
       context precision, context recall, faithfulness, answer relevancy
    4. write the scores to frontend/assets/eval_results.json -> the UI chart

Run:
    python backend/evaluate.py

WARNING: this is CPU-heavy. RAGAS makes many LLM calls (the judge), so on a
local model this can take a long time and will load the CPU hard. Consider an
API judge (see JUDGE below) for speed and reliability — small local models can
return malformed output that RAGAS scores as NaN.

Requires (not in the core install):  pip install ragas datasets langchain-ollama
"""

import json
import os

import config
import pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(_ROOT, "data", "eval_set.json")
RESULTS = os.path.join(_ROOT, "frontend", "assets", "eval_results.json")

# Which models judge the metrics. "ollama" = local & free (slow, can be noisy);
# "openai" = reliable & fast, needs OPENAI_API_KEY.
JUDGE = "ollama"

# The ablation configs, as (label, USE_BM25, USE_RERANKER, USE_ITERATIVE).
CONFIGS = [
    ("Semantic", False, False, False),
    ("+ BM25", True, False, False),
    ("+ Reranker", True, True, False),
    ("+ Iterativ", True, True, True),
]

# RAGAS metric -> friendly Romanian label used in the chart.
METRIC_LABELS = {
    "context_precision": "Precizie context",
    "context_recall": "Acoperire context",
    "faithfulness": "Fidelitate",
    "answer_relevancy": "Relevanță răspuns",
}


def _judge():
    """Return (llm, embeddings) wrapped for RAGAS."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    if JUDGE == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return (
            LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini")),
            LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small")),
        )
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    return (
        LangchainLLMWrapper(ChatOllama(model=config.LLM_MODEL, temperature=0)),
        LangchainEmbeddingsWrapper(OllamaEmbeddings(model=config.LLM_MODEL)),
    )


def run_config(label, use_bm25, use_reranker, use_iterative, test_set):
    """Run the pipeline for one config over the whole test set."""
    config.USE_BM25 = use_bm25
    config.USE_RERANKER = use_reranker
    config.USE_ITERATIVE = use_iterative
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, item in enumerate(test_set, 1):
        print(f"  [{label}] {i}/{len(test_set)}: {item['question'][:50]}...")
        out = pipeline.answer_question(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(out["answer"])
        rows["contexts"].append([c["text"] for c in out["chunks"]])
        rows["ground_truth"].append(item["ground_truth"])
    return rows


def score(rows, llm, emb):
    """Score one config's rows with RAGAS -> {metric_key: mean float}."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        context_precision, context_recall, faithfulness, answer_relevancy,
    )
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    result = evaluate(Dataset.from_dict(rows), metrics=metrics, llm=llm, embeddings=emb)
    df = result.to_pandas()
    return {m.name: float(df[m.name].mean()) for m in metrics}


def main():
    with open(EVAL_SET, encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"Set de test: {len(test_set)} întrebări. Judecător: {JUDGE}.")

    llm, emb = _judge()
    per_config = {}
    for label, bm25, rer, itr in CONFIGS:
        rows = run_config(label, bm25, rer, itr, test_set)
        print(f"  Punctez cu RAGAS [{label}]...")
        per_config[label] = score(rows, llm, emb)

    # Reshape into the chart format: one entry per metric, values per config.
    metrics_out = []
    for key, romanian in METRIC_LABELS.items():
        metrics_out.append({
            "key": romanian,
            "vals": [round(per_config[label].get(key, 0.0), 3) for label, *_ in CONFIGS],
        })

    out = {
        "generated": True,
        "n_questions": len(test_set),
        "configs": [label for label, *_ in CONFIGS],
        "metrics": metrics_out,
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Gata. Scoruri scrise în {RESULTS}. Deschide tab-ul „Evaluează” în UI.")


if __name__ == "__main__":
    main()
