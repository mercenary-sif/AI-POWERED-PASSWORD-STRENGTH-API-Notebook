from pydantic import BaseModel, field_validator

class PasswordRequest(BaseModel):
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Password cannot be empty.")
        if len(v) > 30:
            # Notebook truncates at MAX_LEN=30 — warn but still process
            pass
        if not v.isprintable():
            raise ValueError("Password must contain printable characters only.")
        return v


class RuleDetail(BaseModel):
    key:   str
    label: str

class Rules(BaseModel):
    passed: list[RuleDetail]
    failed: list[RuleDetail]

class CharacterCounts(BaseModel):
    letters:       int
    uppercase:     int
    lowercase:     int
    digits:        int
    special_chars: int

class ConfidenceScores(BaseModel):
    weak:   float
    medium: float
    strong: float

class PasswordResponse(BaseModel):
    password_length:   int
    strength:          str          # "weak" | "medium" | "strong"
    confidence:        float        # winning class probability %
    confidence_scores: ConfidenceScores
    entropy:           float
    character_counts:  CharacterCounts
    rules:             Rules
    suggestions:       list[str]