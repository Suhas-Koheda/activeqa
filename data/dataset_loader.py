"""
Dataset loading utilities for TyDiQA (Telugu) and IndicQA.

Expected output format (list of dicts):
    {
        "query":       str,
        "language":    str,
        "gold_answer": str,
        "doc_ids":     list[str]      # IDs of gold-passage documents
    }
"""
from __future__ import annotations
import json
import os
from datasets import load_dataset
from config import TYDYQA_LANG, INDICQA_LANG, MAX_QUERIES


def load_tydiqa(split: str = "train") -> list[dict]:
    """
    Load TyDiQA-GoldP for Telugu and flatten to query-level records.
    """
    ds = load_dataset("google-research-datasets/tydiqa", "primary_task", split=split)
    records = []
    for item in ds:
        # TyDiQA passage-task fields
        query     = item.get("question", "")
        lang      = TYDYQA_LANG
        answers   = item.get("answers", [])
        gold_text = answers[0]["text"] if answers else ""
        # doc id for retrieval ground truth — use title or passage id
        title  = item.get("title", "")
        doc_id = title.strip().lower().replace(" ", "_") if title else item.get("passage_id", "")
        records.append({
            "query":       query,
            "language":    lang,
            "gold_answer": gold_text,
            "doc_ids":     [doc_id] if doc_id else [],
        })
    if MAX_QUERIES:
        records = records[:MAX_QUERIES]
    return records


def load_indicqa(split: str = "train") -> list[dict]:
    """
    Load IndicQA (Telugu subset) from ai4bharat/IndicQA.
    Falls back gracefully if dataset schema differs.
    """
    try:
        ds = load_dataset("ai4bharat/IndicQA", INDICQA_LANG, split=split)
    except Exception:
        # Fallback: try without subset name
        ds = load_dataset("ai4bharat/IndicQA", split=split)

    records = []
    for item in ds:
        query     = item.get("question", item.get("query", ""))
        answers   = item.get("answers", item.get("answer", []))
        if isinstance(answers, list):
            gold_text = answers[0] if answers else ""
        else:
            gold_text = str(answers)
        doc_id = item.get("doc_id", item.get("passage_id", ""))
        records.append({
            "query":       query,
            "language":    INDICQA_LANG,
            "gold_answer": gold_text,
            "doc_ids":     [doc_id] if doc_id else [],
        })
    if MAX_QUERIES:
        records = records[:MAX_QUERIES]
    return records


def save_records(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
