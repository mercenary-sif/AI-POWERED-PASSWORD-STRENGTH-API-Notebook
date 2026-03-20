# ============================================================
# OLD VERSION (tensorflow) — kept for reference
# ============================================================

# import tensorflow as tf
# from pathlib import Path
# from app.utils.preprocessing import load_vocab

# BASE_DIR = Path(__file__).resolve().parent.parent.parent

# MODEL_PATH = BASE_DIR / "saved_model" / "bilstm_password_model (1).keras"
# MODEL_PATH = BASE_DIR / "saved_model" / "bilstm_password_model.onnx"
# VOCAB_PATH = BASE_DIR / "saved_model" / "char_vocab.pkl"

# _model = None
# _char_vocab = None

# def get_model() -> tf.keras.Model:
#     global _model
#     if _model is None:
#         print("Loading ML model...")
#         _model = tf.keras.models.load_model(MODEL_PATH)
#     return _model

# def get_vocab() -> dict:
#     global _char_vocab
#     if _char_vocab is None:
#         print("Loading character vocabulary...")
#         _char_vocab = load_vocab(VOCAB_PATH)
#     return _char_vocab

# ============================================================
# NEW VERSION (onnxruntime) — Python 3.12 compatible
# ============================================================

import onnxruntime as ort
import pickle
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "saved_model" / "bilstm_password_model.onnx"
VOCAB_PATH = BASE_DIR / "saved_model" / "char_vocab.pkl"

_session    = None
_char_vocab = None


def get_model() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"❌ ONNX model not found at {MODEL_PATH}")
        print(f"📦 Loading ONNX model from {MODEL_PATH}...")
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        print("✅ ONNX model loaded")
    return _session


def get_vocab() -> dict:
    global _char_vocab
    if _char_vocab is None:
        if not VOCAB_PATH.exists():
            raise RuntimeError(f"❌ Vocab not found at {VOCAB_PATH}")
        print(f"📖 Loading vocabulary from {VOCAB_PATH}...")
        with open(VOCAB_PATH, "rb") as f:
            _char_vocab = pickle.load(f)
        print("✅ Vocabulary loaded")
    return _char_vocab