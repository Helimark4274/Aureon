"""Optional FastAPI wrapper. Run: uvicorn aureon.api:app   (note: 'aureon.api:app', not 'main.py:app')."""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import Aureon
from .sim import make_generator

ORIGINS = os.environ.get("AUREON_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app = FastAPI(title="Aureon", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

def _build_engine() -> Aureon:
    """AUREON_BACKEND=anthropic -> real model (needs ANTHROPIC_API_KEY + `pip install anthropic`).
    Anything else -> synthetic simulator (used by the tests)."""
    if os.environ.get("AUREON_BACKEND", "sim") == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("AUREON_BACKEND=anthropic but ANTHROPIC_API_KEY is not set")
        from .llm_anthropic import anthropic_generator
        return Aureon(anthropic_generator(model=os.environ.get("AUREON_MODEL", "claude-sonnet-4-6")))
    return Aureon(make_generator(seed=0))


engine = _build_engine()


class AskIn(BaseModel):
    query: str
    context: Dict[str, Any] = {}


class FeedbackIn(BaseModel):
    decision_id: int
    correct: bool


@app.get("/api/status")
def status():
    return {"weights": engine.weights.w, "calibrator": {"a": engine.calibrator.a, "b": engine.calibrator.b,
            "fitted": engine.calibrator.fitted}, "labelled": len(engine.history),
            "blind_spots": [r for r in engine.detector.report() if r["flagged"]]}


@app.post("/api/ask")
def ask(body: AskIn):
    d = engine.answer(body.query, body.context)
    return {k: v for k, v in d.__dict__.items() if not k.startswith("_")}


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    engine.feedback(body.decision_id, body.correct)
    return {"ok": True}


@app.post("/api/calibrate")
def calibrate():
    engine.fit_calibration()
    return {"a": engine.calibrator.a, "b": engine.calibrator.b, "fitted": engine.calibrator.fitted}


@app.get("/api/gaps")
def gaps():
    """Where Aureon abstained, grouped by slice and reason: a to-do list of what to label next."""
    return [{"slice": a, "reason_code": b, "count": c} for a, b, c in engine.store.gaps()]
