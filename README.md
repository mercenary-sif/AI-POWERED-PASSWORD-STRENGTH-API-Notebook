# 🔐 MS-AI Password Strength API

> A production-ready REST API serving a **BiLSTM deep learning model** that classifies password strength as Weak, Medium, or Strong — trained end-to-end on ~850K real, synthetic, and adversarial passwords using a full ML pipeline built in Google Colab.

---

## Table of Contents

- [Why This Is Different](#-why-this-is-different)
- [Project Architecture](#-project-architecture)
- [Dataset Pipeline](#-dataset-pipeline)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Run Locally](#-run-locally)
- [Deployment on Render](#-deployment-on-render-free)
- [Production Notes](#-production-notes)
- [Testing](#-testing)

---

## 🎯 Why This Is Different

Most password checkers apply **static rules**: length >= 8, has a digit, has a symbol. A password like `Password@1` passes every one of those checks — yet any security researcher will tell you it is trivially weak.

MS-AI takes a fundamentally different approach. Instead of checking rules, the model **learned what real attackers already know** by training on:

- Actual passwords from the **RockYou breach** (real human behavior)
- **Adversarially crafted** samples designed to fool rule-based systems
- **Leet speak** variants (`p@$$w0rd`, `dr@g0n`) — the model was explicitly trained to catch these
- **Keyboard walks** (`Qwerty@1`, `1qaz2wsx`) — pattern-detected at character level
- **Name + year combos** (`Ahmed2024!`, `@Karim2025`) — social engineering patterns
- **Decorated repeats** (`Aaaa@1234`, `Zzzz!2024`) — zero entropy despite looking complex

The result: a BiLSTM that reads passwords **character by character**, the same way a language model understands text, and classifies strength based on learned sequence patterns not a rule-book.

---

## 🏗 Project Architecture

```
Google Colab (Training)
        |
        |-- Data Collection & Generation
        |       |-- RockYou dataset (Kaggle)         333K samples
        |       |-- Rule-based generator             350K samples
        |       |-- Adversarial edge-case generator  150K samples
        |
        |-- Data Cleaning & Merging
        |       |-- Deduplication (global)
        |       |-- Label conflict resolution
        |       |-- Final: clean_password_dataset.csv
        |
        |-- Preprocessing
        |       |-- Stratified 70/15/15 split
        |       |-- Character-level tokenization
        |       |-- Padding to MAX_LEN=30
        |       |-- Oversampling minority classes
        |       |-- One-hot label encoding
        |
        |-- BiLSTM Training
                |-- Embedding(vocab_size, 64)
                |-- Bidirectional LSTM(128)
                |-- Dense(64, relu) + Dropout
                |-- Dense(3, softmax)
                |-- Saved: bilstm_password_model.keras
                |-- Saved: char_vocab.pkl

FastAPI Backend (Production)
        |
        |-- Startup: load model + vocab (singleton, once)
        |-- POST /analyze
                |-- Pydantic validation
                |-- Character tokenization (same vocab)
                |-- Pad to MAX_LEN=30
                |-- BiLSTM inference -> softmax probs
                |-- Rule engine (10+ policy checks)
                |-- Shannon entropy
                |-- Suggestions builder
                |-- Structured JSON response
```

---

## 📊 Dataset Pipeline

The training data was built from three complementary sources, each serving a specific purpose in teaching the model different aspects of password strength.

### Source 1 — Rule-Based Generated (350K passwords)

Generated using a cryptographically secure generator (`secrets` module) with explicit policy rules:

| Class | Policy | Count |
|---|---|---|
| Weak (0) | Length 4-7, lowercase/digits only, common patterns | ~116K |
| Medium (1) | Length 8-11, upper+lower+digits, no special chars | ~116K |
| Strong (2) | Length 12-20, all char types, min 2 of each | ~116K |

Weak generation strategies: common words, name+year combos, digit-only strings, repeated chars, keyboard sequences.

Strong generation: guaranteed minimum of 2 uppercase + 2 lowercase + 2 digits + 2 special chars before filling remaining length from the full charset.

Saved to: `rule_based_350k.csv`

### Source 2 — Adversarial Edge Cases (150K passwords)

Specifically designed to train the model on passwords that **fool naive rule-based checkers**:

| Strategy | Label | What It Teaches |
|---|---|---|
| `deceptive_weak` | 0 (Weak) | Common word + case mix + leet + suffix looks strong |
| `full_leet_weak` | 0 (Weak) | Full leet substitution on common word |
| `keyboard_walk` | 0 (Weak) | `Qwerty@1`, `1qaz2wsx123!` pass naive rules |
| `name_year_combo` | 0 (Weak) | `Ahmed2024!`, `@Karim2025` — social patterns |
| `repeat_pattern_weak` | 0 (Weak) | `Aaaa@1234` — zero entropy despite decoration |
| `prefix_word_weak` | 0 (Weak) | `MyDragon99!` — still word-based |
| `disguised_medium` | 1 (Medium) | Short (4-6 chars) but all char types present |
| `medium_no_special` | 1 (Medium) | Good length + upper + digit but missing special |
| `truly_strong` | 2 (Strong) | High-entropy base + leet + double special insert |
| `passphrase_strong` | 2 (Strong) | Multi-word passphrase with substitutions |

Mutations applied: `apply_leet()`, `apply_full_leet()`, `random_case_mix()`, `random_special_insert()`, `inject_digit_block()`, `add_common_suffix()`, `add_common_prefix()`

Saved to: `adversarial_250k.csv`

### Source 3 — RockYou Real Breach Data (333K passwords)

Downloaded from Kaggle (`wjburns/common-password-list-rockyoutxt`). Applied cleaning pipeline:

- Stripped whitespace, removed empty strings
- Filtered length: kept passwords between 4 and 30 characters
- Removed non-printable and whitespace-containing entries
- Global deduplication via hash set

**Labeling policy** (mirrors training labeler exactly):

| Rule | Check |
|---|---|
| R1 | length >= 8 |
| R2 | has uppercase |
| R3 | has lowercase |
| R4 | has digit |
| R5 | has special character |
| R6 | not in blacklist (top 15 common passwords) |
| R7 | no sequential pattern (abc, 123, zyx) |
| R8 | length >= 12 (strong only) |

Strong = all 8 rules pass. Medium = 4+ rules pass AND R1 passes. Weak = everything else.

**Key design decision:** Strong passwords were intentionally excluded from the RockYou sample. Real users almost never create genuinely strong passwords. Including mislabeled "strong" RockYou samples would add label noise. The strong class is entirely covered by the rule-based and adversarial sources.

Stratified sample: 270K weak + 63K medium + 0 strong = 333K

Saved to: `rockyou_333k.csv`

### Dataset Merging & Deduplication

All three sources were merged and cleaned through a rigorous pipeline:

```
Step 1: Concatenate all 3 sources
        -> Total before dedup: ~833K rows

Step 2: Internal duplicate audit (per source)
        -> audit_internal_duplicates() per dataset

Step 3: Cross-dataset duplicate detection
        -> df_merged.duplicated("password", keep=False)
        -> Identify passwords appearing in multiple sources

Step 4: Label conflict resolution
        -> Passwords with different strength labels across sources
        -> Creates label noise -> REMOVED entirely

Step 5: Global deduplication
        -> drop_duplicates(subset="password")
        -> keep first occurrence

Step 6: Shuffle
        -> sample(frac=1, random_state=42)

Final: clean_password_dataset.csv
```

---

## ⚙️ Preprocessing Pipeline

Applied before BiLSTM training on `clean_password_dataset.csv`:

### 1. Dataset Inspection

```
Shape, column types, null check, duplicate check
Label distribution (pie chart)
Source distribution (bar chart)
Password length distribution (histogram with 90th/95th percentile markers)
```

### 2. Stratified Train / Validation / Test Split

```
70% Train    -> X_train, y_train
15% Val      -> X_val,   y_val      (stratify=y preserves class ratios)
15% Test     -> X_test,  y_test
```

### 3. Character-Level Tokenization

Vocabulary built from training set only (no leakage into val/test):

```
Special tokens:
  <PAD> -> index 0   (padding token)
  <UNK> -> index 1   (unseen characters in val/test)

All other chars: sorted and assigned indices starting at 2

char_to_idx saved as: char_vocab.pkl (used at inference time in API)

Tokenization: each character mapped to its integer index
Unknown chars (not seen in training): mapped to UNK_TOKEN (1)
```

### 4. Padding

```
MAX_LEN = 30         (covers 95th+ percentile of password lengths)
padding  = 'post'    (zeros appended at end)
truncating = 'post'  (long passwords cut from end)

Output shapes:
  X_train_seq: (N_train, 30)
  X_val_seq:   (N_val,   30)
  X_test_seq:  (N_test,  30)
```

Attention masks generated: `(X_seq != PAD_TOKEN).astype(float32)` — used by `mask_zero=True` in the Embedding layer to ignore padding during LSTM computation.

### 5. Class Imbalance Handling

Two-pronged approach:

**Oversampling** — minority classes resampled to match majority class count using `sklearn.utils.resample(replace=True)`:

```
df_medium_up = resample(df_medium, n_samples=len(df_weak), replace=True)
df_strong_up = resample(df_strong, n_samples=len(df_weak), replace=True)
df_train_bal = concat([df_weak, df_medium_up, df_strong_up]).shuffle()
```

**Class weights** — computed via `compute_class_weight('balanced')` and passed to `model.fit(class_weight=CLASS_WEIGHTS)` as a second safeguard.

### 6. One-Hot Label Encoding

```python
y_train_cat = to_categorical(y_train_bal, num_classes=3)
# shape: (N_train_balanced, 3)
# e.g. label=2 -> [0, 0, 1]
```

Saved to: `preprocessed_balanced_data.npz` (compressed numpy archive)

---

## 🧠 Model Architecture

```
Input: (batch_size, 30)  <- padded integer sequences
  |
  Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True)
  -> (batch_size, 30, 64)   <- char embeddings, PAD tokens masked
  |
  Bidirectional(LSTM(128, return_sequences=False,
                     dropout=0.3, recurrent_dropout=0.3))
  -> (batch_size, 256)      <- 128 forward + 128 backward
  |
  Dense(64, activation='relu')
  |
  Dropout(0.3)
  |
  Dense(3, activation='softmax')
  -> (batch_size, 3)        <- [P(weak), P(medium), P(strong)]
```

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 64 |
| BiLSTM units | 128 (x2 = 256 total) |
| Dropout | 0.3 (both LSTM and recurrent) |
| Output activation | Softmax |
| Loss | Categorical crossentropy |
| Optimizer | Adam (lr=1e-3) |
| Batch size | 128 |
| Max epochs | 15 |
| Early stopping | patience=3 on val_loss |

Key design choices:

- `mask_zero=True` in Embedding: LSTM ignores PAD tokens entirely — padding does not contribute to gradient updates
- `return_sequences=False`: only the final hidden state is passed to Dense layers
- Bidirectional wrapper: reads password both left-to-right AND right-to-left — captures suffix patterns (`@1`, `123!`) and prefix patterns (`My`, `Dr`) simultaneously
- Recurrent dropout applied inside LSTM cells for stronger regularization

---

## 🏋️ Training Pipeline

```python
# Callbacks used
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ModelCheckpoint(save_best_only=True, monitor='val_loss')
CSVLogger(bilstm_training_log.csv)

# Training call
model.fit(
    X_train_seq_bal, y_train_cat_bal,
    validation_data=(X_val_seq, y_val_cat),
    epochs=15,
    batch_size=128,
    class_weight=CLASS_WEIGHTS,   # balanced after oversampling
    callbacks=[early_stop, checkpoint, csv_logger]
)
```

Saved artifacts:
- `bilstm_best_model.h5` — best checkpoint (by val_loss)
- `bilstm_password_model.keras` — final saved model
- `char_vocab.pkl` — character-to-index mapping
- `bilstm_training_log.csv` — epoch-by-epoch metrics

Evaluation on held-out test set:
```python
test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, batch_size=128)
```

---

## 📡 API Reference

### GET /health

```http
GET /health
```

Response `200`:
```json
{
  "status": "ok",
  "model": "BiLSTM",
  "max_len": 30
}
```

---

### POST /analyze

Runs a password through the BiLSTM model and rule engine. Returns class prediction, confidence scores, entropy, character breakdown, rule audit, and suggestions.

```http
POST /analyze
Content-Type: application/json
```

Request:
```json
{
  "password": "Example123!"
}
```

Full Response `200`:
```json
{
  "password_length": 11,
  "strength": "medium",
  "confidence": 87.52,
  "confidence_scores": {
    "weak":   5.31,
    "medium": 87.52,
    "strong": 7.17
  },
  "entropy": 3.4594,
  "character_counts": {
    "letters":       7,
    "uppercase":     1,
    "lowercase":     6,
    "digits":        3,
    "special_chars": 1
  },
  "rules": {
    "passed": [
      { "key": "has_uppercase",   "label": "Contains uppercase letter" },
      { "key": "has_digit",       "label": "Contains digit" },
      { "key": "has_special",     "label": "Contains special character" },
      { "key": "not_blacklisted", "label": "Not in common password list" }
    ],
    "failed": [
      { "key": "min_12_chars",       "label": "At least 12 characters" },
      { "key": "two_plus_specials",  "label": "At least 2 special characters" },
      { "key": "two_plus_uppercase", "label": "At least 2 uppercase letters" }
    ]
  },
  "suggestions": [
    "Extend to 12+ characters to reach Strong — every extra character multiplies attack time exponentially.",
    "Include at least 2 special characters from: !@#$%^&*()_+-=[]{}|;:,.<>?",
    "Add at least 2 uppercase letters."
  ]
}
```

Error `422`:
```json
{
  "error": true,
  "message": "Password cannot be empty.",
  "code": "VALIDATION_ERROR"
}
```

Error `500`:
```json
{
  "error": true,
  "message": "Internal server error during prediction.",
  "code": "SERVER_ERROR"
}
```

**How inference works at the API level:**

```
POST body received
      |
Pydantic validation    <- rejects empty, wrong types
      |
preprocess(password)   <- same tokenize() + pad_sequences() used in training
      |                   char_to_idx from char_vocab.pkl, MAX_LEN=30, UNK=1
model.predict(seq)     <- softmax [P_weak, P_medium, P_strong]
      |
analyze_rules()        <- 10+ rule checks mirroring label_password() from notebook
      |
password_entropy()     <- Shannon entropy: -sum(p * log2(p))
      |
build_suggestions()    <- rule-aware + label-aware guidance list
      |
JSON response
```

**Critical consistency note:** The `preprocess()` function in `app/utils/preprocessing.py` implements the exact same tokenization logic as cells 80-84 in the training notebook — same `char_to_idx` vocab, same `MAX_LEN=30`, same `PAD=0`, same `UNK=1`. Any drift between training preprocessing and inference preprocessing causes silent accuracy degradation.

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                    <- FastAPI app, lifespan startup, routes, error handlers
│   ├── schemas/
│   │   └── password_schema.py     <- Pydantic PasswordRequest / PasswordResponse models
│   ├── model/
│   │   ├── model_loader.py        <- Singleton: loads .keras model + char_vocab.pkl once
│   │   └── predictor.py           <- predict(), build_suggestions(), password_entropy()
│   └── utils/
│       └── preprocessing.py       <- preprocess(), analyze_rules(), BLACKLIST, SPECIAL_CHARS
├── models/
│   └── bilstm_password_model.keras   <- Trained Keras model
├── vocab/
│   └── char_vocab.pkl                <- char_to_idx dict (saved with pickle)
├── requirements.txt
├── Procfile                          <- Render: web: uvicorn app.main:app ...
└── README.md
```

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.10+
- `models/bilstm_password_model.keras` — trained model
- `vocab/char_vocab.pkl` — character vocabulary

### Install

```bash
# Clone the repository
git clone https://github.com/your-username/ms-ai-password-api.git
cd ms-ai-password-api/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
tensorflow>=2.15.0
numpy>=1.26.0
pydantic>=2.6.0
```

---

## ▶️ Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/analyze` | Main inference endpoint |
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc documentation |

---

## 🧪 Testing

**cURL — key test cases:**

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Deceptive weak — passes all static rules, model flags as weak
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"password": "Password@1"}'

# 3. Leet speak on common word — still weak
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"password": "p@$$w0rd123!"}'

# 4. Name + year — social pattern flagged
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"password": "Ahmed2024!"}'

# 5. Genuinely strong password
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"password": "Qu@n7umL3@pWU.88TSG"}'

# 6. Empty input — should return 422
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"password": ""}'
```
---

## 👨‍💻 Author

Built as a complete end-to-end ML engineering project:


