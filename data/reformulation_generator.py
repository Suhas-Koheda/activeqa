"""
Reformulation candidate generation using IndicBART.

Given a query, produce one candidate reformulation per action:
    KEEP          → original query
    NORMALIZE     → lowercased, punctuation-stripped
    EXPAND        → IndicBART with expansion prompt
    TRANSLITERATE → Telugu ↔ Roman Telugu (rule-based)
    TRANSLATE     → IndicBART translate prompt (Telugu→English)
    BILINGUALIZE  → original + English translation concatenated
"""
from __future__ import annotations
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import BART_MODEL, HF_TOKEN, GEN_MAX_LEN, GEN_NUM_BEAMS, GEN_BATCH_SIZE


class ReformulationGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"Loading IndicBART ({BART_MODEL}) on {device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(BART_MODEL, use_auth_token=HF_TOKEN)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            BART_MODEL, use_auth_token=HF_TOKEN
        ).to(device)
        self.model.eval()

    # ── Action helpers ───────────────────────────────────────────────────────

    def keep(self, query: str) -> str:
        return query

    def normalize(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"[^\w\s]", "", q)
        q = re.sub(r"\s+", " ", q)
        return q

    def transliterate(self, query: str) -> str:
        """
        Very simple Telugu ↔ Roman Telugu mapping.
        For production, swap with a proper library like indic-transliteration.
        """
        # If query is already in Telugu script, romanize
        if any("\u0C00" <= ch <= "\u0C7F" for ch in query):
            return self._telugu_to_roman(query)
        return query  # assume already roman

    def _telugu_to_roman(self, text: str) -> str:
        mapping = {
            "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
            "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au",
            "క": "ka", "ఖ": "kha", "గ": "ga", "ఘ": "gha", "ఙ": "nga",
            "చ": "ca", "ఛ": "cha", "జ": "ja", "ఝ": "jha", "ఞ": "nya",
            "ట": "tta", "ఠ": "ttha", "డ": "dda", "ఢ": "ddha", "ణ": "nna",
            "త": "ta", "థ": "tha", "ద": "da", "ధ": "dha", "న": "na",
            "ప": "pa", "ఫ": "pha", "బ": "ba", "భ": "bha", "మ": "ma",
            "య": "ya", "ర": "ra", "ల": "la", "వ": "va", "శ": "sha",
            "ష": "ssa", "స": "sa", "హ": "ha", "ళ": "lla", "ఱ": "rra",
            "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu",
            "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
            "ం": "m", "ః": "h",
        }
        result = []
        for ch in text:
            result.append(mapping.get(ch, ch))
        return "".join(result)

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=GEN_MAX_LEN).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=GEN_MAX_LEN,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def expand(self, query: str) -> str:
        prompt = f"Expand this Telugu question with synonyms and context: {query}"
        return self._generate(prompt)

    def translate(self, query: str) -> str:
        prompt = f"Translate from Telugu to English: {query}"
        return self._generate(prompt)

    def bilingualize(self, query: str) -> str:
        en = self.translate(query)
        return f"{query} {en}"

    # ── Full pipeline ────────────────────────────────────────────────────────

    def generate_all(self, query: str) -> dict[str, str]:
        """Return a dict mapping action name → reformulated query."""
        return {
            "KEEP":          self.keep(query),
            "NORMALIZE":     self.normalize(query),
            "EXPAND":        self.expand(query),
            "TRANSLITERATE": self.transliterate(query),
            "TRANSLATE":     self.translate(query),
            "BILINGUALIZE":  self.bilingualize(query),
        }
