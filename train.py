"""
Main training script for the Active QA Query Reformulation RL agent.

Pipeline:
    1. Load dataset (TyDiQA / IndicQA)
    2. Build document corpus
    3. Generate reformulation candidates (all 6 actions)
    4. Evaluate retrieval for each action → reward matrix
    5. Train LinUCB and Neural Bandit
    6. Save agents

Usage:
    python train.py --dataset tydiqa --agent both
    python train.py --dataset indicqa --agent linucb
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import (ACTIONS, N_ACTIONS, TOP_K, ALPHA, BETA, GAMMA, LAMBDA,
                    BANDIT_EPOCHS, BANDIT_LR, TRAIN_BATCH, CHECKPOINT_DIR,
                    MAX_QUERIES)
from data.dataset_loader import load_tydiqa, load_indicqa, save_records
from data.reformulation_generator import ReformulationGenerator
from retrieval.retriever import BM25Retriever
from retrieval.metrics import recall_at_k, mrr_at_k, f1_score
from agent.features import QueryFeatureExtractor
from agent.bandit import LinUCB, train_linucb, train_neural_bandit
from agent.policy_network import NeuralBandit


def build_corpus(records: list[dict]) -> dict[str, str]:
    """Build {doc_id: text} corpus from records."""
    corpus = {}
    for rec in records:
        for doc_id in rec.get("doc_ids", []):
            corpus[doc_id] = rec.get("gold_answer", "")  # placeholder: use doc text if available
    return corpus


def compute_rewards(records: list[dict], reformulator, retriever) -> np.ndarray:
    """
    For every record and every action, compute the reward:
        R = alpha * delta_Recall@k + beta * delta_MRR + gamma * delta_F1 - lambda * cost
    Returns an (N, N_ACTIONS) reward matrix.
    """
    rewards = np.zeros((len(records), N_ACTIONS), dtype=np.float32)
    print("Computing reward matrix …")
    for i, rec in enumerate(tqdm(records)):
        query    = rec["query"]
        relevant = rec.get("doc_ids", [])

        # Original retrieval
        orig_retrieved = retriever.retrieve(query, k=TOP_K)
        orig_recall    = recall_at_k(orig_retrieved, relevant, TOP_K)
        orig_mrr       = mrr_at_k(orig_retrieved, relevant, TOP_K)

        for j, action in enumerate(ACTIONS):
            if action == "KEEP":
                reformed = query
                cost     = 0.0
            elif action == "NORMALIZE":
                reformed = reformulator.normalize(query)
                cost     = 0.01
            elif action == "EXPAND":
                reformed = reformulator.expand(query)
                cost     = 0.1
            elif action == "TRANSLITERATE":
                reformed = reformulator.transliterate(query)
                cost     = 0.05
            elif action == "TRANSLATE":
                reformed = reformulator.translate(query)
                cost     = 0.1
            elif action == "BILINGUALIZE":
                reformed = reformulator.bilingualize(query)
                cost     = 0.15
            else:
                continue

            retrieved = retriever.retrieve(reformed, k=TOP_K)
            recall    = recall_at_k(retrieved, relevant, TOP_K)
            mrr       = mrr_at_k(retrieved, relevant, TOP_K)

            delta_recall = recall - orig_recall
            delta_mrr    = mrr    - orig_mrr
            reward = ALPHA * delta_recall + BETA * delta_mrr - LAMBDA * cost
            rewards[i, j] = reward

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Train Active QA reformulation agent")
    parser.add_argument("--dataset", choices=["tydiqa", "indicqa"], default="tydiqa")
    parser.add_argument("--agent",   choices=["linucb", "neural", "both"], default="both")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"Loading {args.dataset} dataset …")
    if args.dataset == "tydiqa":
        records = load_tydiqa("train")
    else:
        records = load_indicqa("train")
    print(f"  Loaded {len(records)} records.")

    # ── 2. Build corpus ──────────────────────────────────────────────────────
    corpus = build_corpus(records)
    print(f"  Corpus size: {len(corpus)} documents.")

    # ── 3. Retriever ─────────────────────────────────────────────────────────
    retriever = BM25Retriever(corpus)

    # ── 4. Reformulation generator ───────────────────────────────────────────
    reformulator = ReformulationGenerator(device=args.device)

    # ── 5. Compute rewards ───────────────────────────────────────────────────
    rewards = compute_rewards(records, reformulator, retriever)
    print(f"  Reward matrix shape: {rewards.shape}")
    print(f"  Mean reward per action: {dict(zip(ACTIONS, rewards.mean(axis=0).round(3)))}")

    # ── 6. Extract states ────────────────────────────────────────────────────
    print("Extracting query features …")
    extractor = QueryFeatureExtractor(device=args.device)
    states = np.stack([extractor.extract(r["query"], r.get("language", "telugu"))
                       for r in records])
    print(f"  State shape: {states.shape}")

    # ── 7. Train agents ──────────────────────────────────────────────────────
    if args.agent in ("linucb", "both"):
        print("\nTraining LinUCB …")
        linucb_agent = train_linucb(states, rewards)
        linucb_path  = f"{CHECKPOINT_DIR}/linucb.pt"
        import pickle
        with open(linucb_path, "wb") as f:
            pickle.dump({"A": linucb_agent.A, "b": linucb_agent.b,
                         "alpha": linucb_agent.alpha, "dim": states.shape[1]}, f)
        print(f"  Saved LinUCB to {linucb_path}")

    if args.agent in ("neural", "both"):
        print("\nTraining Neural Bandit …")
        neural_agent = NeuralBandit(input_dim=states.shape[1], lr=BANDIT_LR, device=args.device)
        history = train_neural_bandit(neural_agent, states, rewards,
                                      epochs=BANDIT_EPOCHS, batch_size=TRAIN_BATCH)
        neural_path = f"{CHECKPOINT_DIR}/neural_bandit.pt"
        neural_agent.save(neural_path)
        print(f"  Saved Neural Bandit to {neural_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
