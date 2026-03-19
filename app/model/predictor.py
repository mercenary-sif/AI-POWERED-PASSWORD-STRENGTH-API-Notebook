import math
import numpy as np
from app.model.model_loader import get_model, get_vocab
from app.utils.preprocessing import preprocess, analyze_rules, BLACKLIST, SPECIAL_CHARS

LABEL_MAP = {0: "weak", 1: "medium", 2: "strong"}

def password_entropy(pwd: str) -> float:
    """Shannon entropy — from notebook Cell 13 / 21."""
    if not pwd:
        return 0.0
    freq = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return round(-sum(p * math.log2(p) for p in freq), 4)

def build_suggestions(pwd: str, rules: dict, label: str) -> list[str]:
    """
    Rule-aware suggestions that guide toward STRONG.
    Based on the exact thresholds from notebook label_password().
    """
    suggestions = []
    r = rules

    if not r["min_8_chars"]["pass"]:
        suggestions.append("Use at least 8 characters — length is the single biggest factor.")

    if not r["min_12_chars"]["pass"]:
        suggestions.append("Extend to 12+ characters to reach Strong — every extra character multiplies attack time exponentially.")

    if not r["has_uppercase"]["pass"] or not r["two_plus_uppercase"]["pass"]:
        suggestions.append("Add at least 2 uppercase letters (e.g. replace 'hello' with 'HeLLo').")

    if not r["has_lowercase"]["pass"]:
        suggestions.append("Include lowercase letters to increase character variety.")

    if not r["has_digit"]["pass"] or not r["two_plus_digits"]["pass"]:
        suggestions.append("Add at least 2 digits spread through the password, not just at the end.")

    if not r["has_special"]["pass"] or not r["two_plus_specials"]["pass"]:
        suggestions.append("Include at least 2 special characters from: !@#$%^&*()_+-=[]{}|;:,.<>?")

    if not r["not_blacklisted"]["pass"]:
        suggestions.append("This password is in the most common breached password list — change it entirely.")

    if not r["no_sequences"]["pass"]:
        suggestions.append("Avoid sequential patterns like 'abc', '123', or 'xyz' — they are trivially guessable.")

    if not r["no_keyboard_pattern"]["pass"]:
        suggestions.append("Avoid keyboard walks like 'qwerty', 'asdfgh', or '1qaz2wsx' — attackers test these first.")

    if not r["no_leet_common_word"]["pass"]:
        suggestions.append("Leet speak on common words (p@ssw0rd, dr@g0n) is well-known to attackers — the model was trained on these patterns.")

    if not r["max_20_chars"]["pass"]:
        suggestions.append("Keep passwords under 20 characters for compatibility with most systems.")

    if label == "weak" and not suggestions:
        suggestions.append("Use a random passphrase of 4+ unrelated words with symbols between them.")

    if label == "medium":
        suggestions.append("You're close to Strong — add more special characters and push length above 12.")

    if label == "strong" and not suggestions:
        suggestions.append("Your password is strong. Make sure you're not reusing it across accounts.")

    return suggestions

def predict(password: str) -> dict:
    model      = get_model()
    char_vocab = get_vocab()

    # ── Preprocessing (notebook Cell 119 exactly) ──────────────
    sequence = preprocess(password, char_vocab)   # shape (1, 30)

    # ── Model inference ─────────────────────────────────────────
    # output: softmax probs [weak, medium, strong] — shape (1, 3)
    probs    = model.predict(sequence, verbose=0)[0]
    class_id = int(np.argmax(probs))
    label    = LABEL_MAP[class_id]

    confidence_scores = {
        "weak":   round(float(probs[0]) * 100, 2),
        "medium": round(float(probs[1]) * 100, 2),
        "strong": round(float(probs[2]) * 100, 2),
    }

    # ── Rule analysis (mirrors label_password from Cell 28) ─────
    analysis = analyze_rules(password)
    rules    = analysis["rules"]
    counts   = analysis["counts"]

    passed_rules = {k: v for k, v in rules.items() if v["pass"]}
    failed_rules = {k: v for k, v in rules.items() if not v["pass"]}

    # ── Suggestions ─────────────────────────────────────────────
    suggestions = build_suggestions(password, rules, label)

    # ── Entropy ─────────────────────────────────────────────────
    entropy = password_entropy(password)

    return {
        "password_length":    counts["length"],
        "strength":           label,
        "confidence":         round(float(probs[class_id]) * 100, 2),
        "confidence_scores":  confidence_scores,
        "entropy":            entropy,
        "character_counts": {
            "letters":        counts["letters"],
            "uppercase":      counts["uppercase"],
            "lowercase":      counts["lowercase"],
            "digits":         counts["digits"],
            "special_chars":  counts["special_chars"],
        },
        "rules": {
            "passed": [{"key": k, "label": v["label"]} for k, v in passed_rules.items()],
            "failed": [{"key": k, "label": v["label"]} for k, v in failed_rules.items()],
        },
        "suggestions": suggestions,
    }