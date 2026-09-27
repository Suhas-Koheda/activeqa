"""
Global configuration for Active QA — Query Reformulation RL Project.
All paths, model IDs, and hyper-parameters are centralized here.
"""
import os

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Model IDs ────────────────────────────────────────────────────────────────
ENCODER_MODEL   = "ai4bharat/indic-bert"          # IndicBERT for query encoding
BART_MODEL      = "ai4bharat/IndicBART"           # IndicBART for reformulation
RETRIEVAL_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"  # DPR-like

# ── Dataset ──────────────────────────────────────────────────────────────────
TYDYQA_LANG     = "telugu"      # primary language
INDICQA_LANG    = "te"          # IndicQA language code
MAX_QUERIES     = 5000          # limit for quick experiments (set None for all)

# ── Reformulation actions ────────────────────────────────────────────────────
ACTIONS = ["KEEP", "NORMALIZE", "EXPAND", "TRANSLITERATE", "TRANSLATE", "BILINGUALIZE"]
N_ACTIONS = len(ACTIONS)

# ── Reward weights ───────────────────────────────────────────────────────────
ALPHA = 1.0     # ΔRecall@k
BETA  = 1.0     # ΔMRR
GAMMA = 0.5     # ΔF1
LAMBDA = 0.1    # reformulation cost penalty

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K       = 5
INDEX_PATH  = "data/faiss_index"

# ── Bandit hyperparameters ──────────────────────────────────────────────────
LINUCB_ALPHA    = 0.5       # exploration parameter for LinUCB
BANDIT_EPOCHS   = 50        # outer training epochs
BANDIT_LR       = 0.01      # learning rate for neural policy
HIDDEN_DIM      = 128       # hidden size of policy network
TRAIN_BATCH     = 64

# ── Generation ───────────────────────────────────────────────────────────────
GEN_MAX_LEN     = 128
GEN_NUM_BEAMS   = 4
GEN_BATCH_SIZE  = 8

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
