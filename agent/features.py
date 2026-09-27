"""
Query feature extraction for the contextual bandit state.
Combines IndicBERT embedding with hand-crafted query features.
"""
from __future__ import annotations
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from config import ENCODER_MODEL, HF_TOKEN


class QueryFeatureExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"Loading IndicBERT encoder ({ENCODER_MODEL}) on {device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, use_auth_token=HF_TOKEN)
        self.model = AutoModel.from_pretrained(ENCODER_MODEL, use_auth_token=HF_TOKEN).to(device)
        self.model.eval()
        self._emb_dim = self.model.config.hidden_size

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=128).to(self.device)
        outputs = self.model(**inputs)
        # Mean-pool last hidden states
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.cpu().numpy()

    def handcrafted_features(self, query: str, language: str = "telugu") -> np.ndarray:
        """
        Returns a small vector of hand-crafted features:
        [query_length, num_words, has_latin_script, has_telugu_script, is_question]
        """
        words = query.split()
        has_latin  = any("a" <= ch.lower() <= "z" for ch in query)
        has_telugu = any("\u0C00" <= ch <= "\u0C7F" for ch in query)
        is_question = float(any(w in query.lower() for w in
                                ["ఏమి", "ఎలా", "ఎప్పుడు", "ఎక్కడ", "వారు", "కాదా",
                                 "what", "how", "when", "where", "who", "?"]))
        return np.array([
            len(query),
            len(words),
            float(has_latin),
            float(has_telugu),
            is_question,
        ], dtype=np.float32)

    def extract(self, query: str, language: str = "telugu") -> np.ndarray:
        """Concatenate IndicBERT embedding + hand-crafted features."""
        emb  = self.encode(query)
        hand = self.handcrafted_features(query, language)
        return np.concatenate([emb, hand]).astype(np.float32)
