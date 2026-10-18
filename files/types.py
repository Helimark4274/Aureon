from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Candidate:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    id: int
    action: str                      # "answer" | "abstain"
    text: Optional[str]
    raw_confidence: float
    confidence: float                # calibrated if a calibrator is fitted
    agreement: float
    threshold: float
    slice: str
    reason: str
    principle_scores: Dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    reason_code: str = ""          # answered | not_learned_yet | low_confidence | no_candidates
    message: str = ""              # human-readable explanation
    labels_in_slice: int = 0
    labels_needed: int = 0
