"""
Baseline strategies for comparison:
    - Always KEEP
    - Always NORMALIZE
    - Always EXPAND
    - Always TRANSLITERATE
    - Always TRANSLATE
    - Always BILINGUALIZE
    - Rule-based selection (heuristic)
"""
from __future__ import annotations
import re
from config import ACTIONS, TOP_K
from retrieval.metrics import recall_at_k, mrr_at_k, f1_score


def rule_based_select(query: str, language: str = "telugu") -> str:
    """
    Simple heuristic to pick an action without learning.
    """
    has_telugu = any("\u0C00" <= ch <= "\u0C7F" for ch in query)
    has_latin  = any("a" <= ch.lower() <= "z" for ch in query)
    words      = query.split()

    # If already English and short → KEEP
    if has_latin and not has_telugu and len(words) < 8:
        return "KEEP"

    # If Telugu and long → EXPAND
    if has_telugu and len(words) > 10:
        return "EXPAND"

    # If mixed script → TRANSLITERATE
    if has_telugu and has_latin:
        return "TRANSLITERATE"

    # If Telugu with spelling issues (repeated chars) → NORMALIZE
    if has_telugu and re.search(r"(.)\1\1", query):
        return "NORMALIZE"

    # Default → BILINGUALIZE
    return "BILINGUALIZE"


def run_baseline(action: str, records: list[dict], reformulator, retriever) -> dict:
    """
    Evaluate a single baseline that always picks the same action.
    """
    metrics = {"recall": [], "mrr": [], "f1": []}
    for rec in records:
        query       = rec["query"]
        relevant    = rec.get("doc_ids", [])
        # Generate reformulation
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
        else:
            raise ValueError(f"Unknown action: {action}")
        # Retrieve
        retrieved = retriever.retrieve(reformed, k=TOP_K)
        metrics["recall"].append(recall_at_k(retrieved, relevant, TOP_K))
        metrics["mrr"].append(mrr_at_k(retrieved, relevant, TOP_K))

    return {
        "action":  action,
        "recall":  sum(metrics["recall"]) / len(metrics["recall"]),
        "mrr":     sum(metrics["mrr"])    / len(metrics["mrr"]),
    }


def run_all_baselines(records: list[dict], reformulator, retriever) -> list[dict]:
    """Run all baselines and return results."""
    results = []
    for action in ACTIONS:
        print(f"  Running baseline: {action} …")
        result = run_baseline(action, records, reformulator, retriever)
        results.append(result)
        print(f"    Recall@{TOP_K}: {result['recall']:.4f}  |  MRR: {result['mrr']:.4f}")

    # Rule-based
    print("  Running baseline: RULE_BASED …")
    rule_metrics = {"recall": [], "mrr": []}
    for rec in records:
        query    = rec["query"]
        relevant = rec.get("doc_ids", [])
        action   = rule_based_select(query, rec.get("language", "telugu"))
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
        rule_metrics["recall"].append(recall_at_k(retrieved, relevant, TOP_K))
        rule_metrics["mrr"].append(mrr_at_k(retrieved, relevant, TOP_K))
    results.append({
        "action": "RULE_BASED",
        "recall": sum(rule_metrics["recall"]) / len(rule_metrics["recall"]),
        "mrr":    sum(rule_metrics["mrr"])    / len(rule_metrics["mrr"]),
    })
    return results
