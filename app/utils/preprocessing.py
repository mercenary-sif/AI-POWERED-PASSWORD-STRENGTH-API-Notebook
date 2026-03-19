import re
import pickle
import numpy as np
from pathlib import Path

MAX_LEN   = 30
PAD_TOKEN = 0
UNK_TOKEN = 1

SPECIAL_CHARS = set("!@#$%^&*()_+-=[]{}|;:,.<>?")

BLACKLIST = {
    "123456", "password", "123456789", "12345678", "12345",
    "1234567", "qwerty", "abc123", "football", "monkey",
    "letmein", "shadow", "master", "dragon", "111111"
}

KEYBOARD_PATTERNS = [
    "qwerty", "asdfgh", "zxcvbn", "1qaz2wsx",
    "1q2w3e4r", "zaq12wsx", "qazwsx", "poiuyt",
    "mnbvcx", "lkjhgf", "0987654321", "qwertyuiop"
]

LEET_MAP = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0',
    's': '$', 't': '7', 'l': '1', 'g': '9', 'b': '8'
}

VOCAB_PATH = Path(__file__).parent.parent.parent / "saved_model" / "char_vocab.pkl"

# Load vocabulary
def load_vocab(path) -> dict:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    return vocab

# Tokenize exactly as in notebook Cell 82 / 119
def tokenize(pwd: str, char_to_idx: dict) -> list[int]:
    return [char_to_idx.get(ch, UNK_TOKEN) for ch in pwd]

# Pad exactly as in notebook Cell 84 / 119
# padding='post', truncating='post', value=PAD_TOKEN
def pad_sequence(seq: list[int]) -> list[int]:
    if len(seq) >= MAX_LEN:
        return seq[:MAX_LEN]
    return seq + [PAD_TOKEN] * (MAX_LEN - len(seq))

# Full preprocessing pipeline → numpy array for model
def preprocess(password: str, char_to_idx: dict) -> np.ndarray:
    password = password.strip()

    seq = tokenize(password, char_to_idx)

    padded = pad_sequence(seq)

    return np.array([padded], dtype=np.int32)

# Rule checks (from notebook label_password / Cell 28)
def has_sequential(pwd: str, n: int = 4) -> bool:
    for i in range(len(pwd) - n + 1):

        chunk = pwd[i:i+n]

        ords = [ord(c) for c in chunk]

        diffs = [ords[j+1] - ords[j] for j in range(len(ords)-1)]

        if all(d == 1 for d in diffs):
            return True

        if all(d == -1 for d in diffs):
            return True

    return False

def has_keyboard_pattern(pwd: str) -> bool:
    p = pwd.lower()

    patterns = KEYBOARD_PATTERNS + [k[::-1] for k in KEYBOARD_PATTERNS]

    return any(pattern in p for pattern in patterns)

def has_leet_on_common_word(pwd: str) -> bool:
    reverse_leet = {v: k for k, v in LEET_MAP.items()}

    decoded = "".join(reverse_leet.get(c, c) for c in pwd.lower())

    return any(word in decoded for word in BLACKLIST)

def analyze_rules(pwd: str) -> dict:
   
    upper_count   = sum(1 for c in pwd if c.isupper())
    lower_count   = sum(1 for c in pwd if c.islower())
    digit_count   = sum(1 for c in pwd if c.isdigit())
    special_count = sum(1 for c in pwd if c in SPECIAL_CHARS)
    letter_count  = upper_count + lower_count

    r1 = len(pwd) >= 8
    r2 = upper_count >= 1
    r3 = lower_count >= 1
    r4 = digit_count >= 1
    r5 = special_count >= 1
    r6 = pwd.lower() not in BLACKLIST
    r7 = not has_sequential(pwd)
    r8 = len(pwd) >= 12                    # needed for STRONG

    # Extra informational flags (not in label_password but useful for UI)
    r9  = not has_keyboard_pattern(pwd)
    r10 = not has_leet_on_common_word(pwd)
    r11 = upper_count >= 2                 # strong requirement
    r12 = digit_count >= 2                 # strong requirement
    r13 = special_count >= 2              # strong requirement
    r14 = len(pwd) <= 20                   # strong upper bound

    rules = {
        "min_8_chars":          {"pass": r1,  "label": "At least 8 characters"},
        "has_uppercase":        {"pass": r2,  "label": "Contains uppercase letter"},
        "has_lowercase":        {"pass": r3,  "label": "Contains lowercase letter"},
        "has_digit":            {"pass": r4,  "label": "Contains a digit"},
        "has_special":          {"pass": r5,  "label": "Contains a special character"},
        "not_blacklisted":      {"pass": r6,  "label": "Not a known common password"},
        "no_sequences":         {"pass": r7,  "label": "No sequential character patterns (abc, 123)"},
        "min_12_chars":         {"pass": r8,  "label": "At least 12 characters (required for Strong)"},
        "no_keyboard_pattern":  {"pass": r9,  "label": "No keyboard walk pattern (qwerty, asdfgh)"},
        "no_leet_common_word":  {"pass": r10, "label": "Not a leet-speak variant of a common word"},
        "two_plus_uppercase":   {"pass": r11, "label": "At least 2 uppercase letters (required for Strong)"},
        "two_plus_digits":      {"pass": r12, "label": "At least 2 digits (required for Strong)"},
        "two_plus_specials":    {"pass": r13, "label": "At least 2 special characters (required for Strong)"},
        "max_20_chars":         {"pass": r14, "label": "At most 20 characters"},
    }

    counts = {
        "length":         len(pwd),
        "uppercase":      upper_count,
        "lowercase":      lower_count,
        "letters":        letter_count,
        "digits":         digit_count,
        "special_chars":  special_count,
    }

    return {"rules": rules, "counts": counts}