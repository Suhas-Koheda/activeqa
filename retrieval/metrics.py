"""
Retrieval and QA evaluation metrics.
"""
from __future__ import annotations
from collections import Counter


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Fraction of relevant docs that appear in top-k retrieved."""
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in relevant if d in topk)
    return hits / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Mean Reciprocal Rank @ k."""
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (used for QA evaluation)."""
    pred_tokens   = Counter(prediction.lower().split())
    gold_tokens   = Counter(ground_truth.lower().split())
    common        = pred_tokens & gold_tokens
    num_same      = sum(common.values())
    if not prediction or not ground_truth:
        return float(prediction == ground_truth)
    if num_same == 0:
        return 0.0
    precision = num_same / sum(pred_tokens.values())
    recall    = num_same / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1.0 if doc in relevant else 0.0
        dcg += rel / (1 if i == 0 else (i).bit_length())  # log2(i+1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / (1 if i == 0 else (i).bit_length()) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
