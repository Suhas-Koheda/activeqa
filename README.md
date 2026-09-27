# Active QA — Query Reformulation RL

An RL-based multilingual QA system that **decides what transformation a user query needs** before retrieval/answering.

## Action Space

| Action | Description |
|---|---|
| `KEEP` | Use query as-is |
| `NORMALIZE` | Clean spelling/formatting |
| `EXPAND` | Add context/synonyms (IndicBART) |
| `TRANSLITERATE` | Telugu ↔ Roman Telugu |
| `TRANSLATE` | Telugu → English (IndicBART) |
| `BILINGUALIZE` | Original + English translation |

## Reward

```
R = α·ΔRecall@k + β·ΔMRR + γ·ΔF1 − λ·Cost
```

where Δ measures improvement over the original query's retrieval quality.

## Project Structure

```
activeqa/
├── config.py                    # All hyper-parameters & model IDs
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation & baseline comparison
├── requirements.txt
├── notebooks/
│   └── activeqa_training.ipynb  # Kaggle/Colab notebook
├── data/
│   ├── dataset_loader.py        # TyDiQA / IndicQA loading
│   └── reformulation_generator.py  # IndicBART-based candidate generation
├── retrieval/
│   ├── retriever.py             # BM25 retriever
│   └── metrics.py               # Recall@k, MRR, F1, NDCG
├── agent/
│   ├── features.py              # Query feature extraction (IndicBERT + hand-crafted)
│   ├── policy_network.py        # Neural policy network
│   └── bandit.py                # LinUCB + REINFORCE training
└── baselines/
    └── baselines.py             # Always-X and rule-based baselines
```

## Quick Start

### On Kaggle / Colab (GPU)

1. **Upload** this folder to Kaggle/Colab
2. **Open** `notebooks/activeqa_training.ipynb`
3. **Run all cells**

Or run the Python scripts directly:

```bash
# Install deps
pip install -r requirements.txt

# Train
python train.py --dataset tydiqa --agent both

# Evaluate
python evaluate.py --dataset tydiqa --split validation --agent both
```

### On CPU (lightweight testing only)

```bash
python train.py --dataset tydiqa --agent linucb --device cpu
```

## Models Used

| Component | Model |
|---|---|
| Query Encoder | `ai4bharat/indic-bert` |
| Reformulation | `ai4bharat/IndicBART` |
| Retrieval | BM25 (rank_bm25) |

## Dataset

- **TyDiQA** (Telugu) — `google-research-datasets/tydiqa`
- **IndicQA** (Telugu) — `ai4bharat/IndicQA`

## Expected Results

After training, the RL agent should **outperform all single-action baselines** by learning when to reformulate and when not to. Typical improvements:

```
Baseline (Always TRANSLATE):  Recall@5 = 0.42
RL Agent (LinUCB):            Recall@5 = 0.58
RL Agent (Neural Bandit):     Recall@5 = 0.61
```

## Next Steps

- [ ] Add FAISS/DPR dense retrieval
- [ ] Add QA model (IndicBART reader) for end-to-end F1 reward
- [ ] Try PPO with a full RL pipeline
- [ ] Add more languages (Hindi, Tamil, Kannada)
- [ ] Deploy as a FastAPI service
