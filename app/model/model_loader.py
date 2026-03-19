import tensorflow as tf
from pathlib import Path
from app.utils.preprocessing import load_vocab

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "saved_model" / "bilstm_password_model (1).keras"
VOCAB_PATH = BASE_DIR / "saved_model" / "char_vocab.pkl"

_model = None
_char_vocab = None


def get_model() -> tf.keras.Model:
    global _model

    if _model is None:
        print("Loading ML model...")
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


def get_vocab() -> dict:
    global _char_vocab

    if _char_vocab is None:
        print("Loading character vocabulary...")
        _char_vocab = load_vocab(VOCAB_PATH)

    return _char_vocab