"""
Evaluation script: compare LinUCB, Neural Bandit, and baselines.

Usage:
    python evaluate.py --dataset tydiqa --split validation --agent both
"""
from __future__ import annotations
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm

from config import (ACTIONS, N_ACTIONS, TOP_K, CHECKPOINT_DIR, RESULTS_DIR, MAX_QUERIES)
from data.dataset_loader import load_tydiqa, load_indicqa
from data.reformulation_generator import ReformulationGenerator
from retrieval.retriever import BM25Retriever
from retrieval.metrics import recall_at_k, mrr_at_k
from agent.features import QueryFeatureExtractor
from agent.bandit import LinUCB
from agent.policy_network import NeuralBandit
from baselines.baselines import run_all_baselines


def build_corpus(records: list[dict]) -> dict[str, str]:
    corpus = {}
    for rec in records:
        for doc_id in rec.get("doc_ids", []):
            corpus[doc_id] = rec.get("gold_answer", "")
    return corpus


def evaluate_linucb(agent: LinUCB, states: np.ndarray,
                    records: list[dict], reformulator, retriever) -> dict:
    recall_list, mrr_list = [], []
    for i, rec in enumerate(records):
        action_idx = agent.select(states[i])
        action     = ACTIONS[action_idx]
        query      = rec["query"]
        relevant   = rec.get("doc_ids", [])
        # Reformulate
        if action == "KEEP":
            reformed = query
        elif action == "NORMALIZE":
            reformed = reformulator.normalize(query)
        elif action == "EXPAND":
            reformed = reformulator.expand(query)
        elif action == "TRANSLITERATE":
            reformed = reformulator.transliterate(query)
        elif action == "TRANSLATE":
            reformed = reformulator.translate(query)
        elif action == "BILINGUALIZE":
            reformed = reformulator.bilingualize(query)
        retrieved = retriever.retrieve(reformed, k=TOP_K)
        recall_list.append(recall_at_k(retrieved, relevant, TOP_K))
        mrr_list.append(mrr_at_k(retrieved, relevant, TOP_K))
    return {"recall": np.mean(recall_list), "mrr": np.mean(mrr_list),
            "agent": "LinUCB"}


def evaluate_neural(agent: NeuralBandit, states: np.ndarray,
                    records: list[dict], reformulator, retriever) -> dict:
    recall_list, mrr_list = [], []
    for i, rec in enumerate(records):
        action_idx = agent.act_greedy(states[i])
        action     = ACTIONS[action_idx]
        query      = rec["query"]
        relevant   = rec.get("doc_ids", [])
        if action == "KEEP":
            reformed = query
        elif action == "NORMALIZE":
            reformed = reformulator.normalize(query)
        elif action == "EXPAND":
            reformed = reformulator.expand(query)
        elif action == "TRANSLITERATE":
            reformed = reformulator.transliterate(query)
        elif action == "TRANSLATE":
            reformed = reformulator.translate(query)
        elif action == "BILINGUALIZE":
            reformed = reformulator.bilingualize(query)
        retrieved = retriever.retrieve(reformed, k=TOP_K)
        recall_list.append(recall_at_k(retrieved, relevant, TOP_K))
        mrr_list.append(mrr_at_k(retrieved, relevant, TOP_K))
    return {"recall": np.mean(recall_list), "mrr": np.mean(mrr_list),
            "agent": "NeuralBandit"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Active QA agents")
    parser.add_argument("--dataset", choices=["tydiqa", "indicqa"], default="tydiqa")
    parser.add_argument("--split",   default="validation")
    parser.add_argument("--agent",   choices=["linucb", "neural", "both"], default="both")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.dataset} ({args.split}) …")
    if args.dataset == "tydiqa":
        records = load_tydiqa(args.split)
    else:
        records = load_indicqa(args.split)
    if MAX_QUERIES:
        records = records[:MAX_QUERIES]
    print(f"  {len(records)} records.")

    corpus    = build_corpus(records)
    retriever = BM25Retriever(corpus)
    reformulator = ReformulationGenerator(device=args.device)

    # Baselines
    print("\n=== Baselines ===")
    baseline_results = run_all_baselines(records, reformulator, retriever)

    # Load agents
    states    = None
    linucb    = None
    neural    = None

    if args.agent in ("linucb", "both"):
        with open(f"{CHECKPOINT_DIR}/linucb.pt", "rb") as f:
            saved = pickle.load(f)
        dim       = saved["dim"]
        linucb    = LinUCB(dim, N_ACTIONS, saved["alpha"])
        linucb.A  = saved["A"]
        linucb.b  = saved["b"]

    if args.agent in ("neural", "both"):
        # Need input_dim — load from feature extractor
        extractor = QueryFeatureExtractor(device=args.device)
        states    = np.stack([extractor.extract(r["query"], r.get("language", "telugu"))
                              for r in records])
        neural    = NeuralBandit(input_dim=states.shape[1], device=args.device)
        neural.load(f"{CHECKPOINT_DIR}/neural_bandit.pt")

    # Evaluate
    print("\n=== RL Agents ===")
    results = list(baseline_results)

    if linucb is not None:
        if states is None:
            extractor = QueryFeatureExtractor(device=args.device)
            states = np.stack([extractor.extract(r["query"], r.get("language", "telugu"))
                               for r in records])
        res = evaluate_linucb(linucb, states, records, reformulator, retriever)
        results.append(res)
        print(f"  LinUCB      — Recall@{TOP_K}: {res['recall']:.4f}  |  MRR: {res['mrr']:.4f}")

    if neural is not None:
        res = evaluate_neural(neural, states, records, reformulator, retriever)
        results.append(res)
        print(f"  NeuralBandit — Recall@{TOP_K}: {res['recall']:.4f}  |  MRR: {res['mrr']:.4f}")

    # Save results
    import json, os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/{args.dataset}_{args.split}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
