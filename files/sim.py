"""Synthetic LLM + question stream with KNOWN per-topic accuracy. Tests Aureon's machinery only."""
import random
from typing import Dict, List, Tuple

from .types import Candidate

TOPIC_ACC: Dict[str, float] = {"arithmetic": 0.92, "geography": 0.85, "history": 0.75, "code": 0.65,
                               "law": 0.45, "medicine": 0.35}


def make_generator(seed: int = 0, topic_acc: Dict[str, float] = None):
    rng = random.Random(seed)
    acc = TOPIC_ACC if topic_acc is None else topic_acc

    def gen(query: str, ctx: dict, n: int) -> List[Candidate]:
        truth, topic = ctx["truth"], ctx["slice"]
        return [Candidate(truth if rng.random() < acc[topic] else f"wrong-{rng.randint(1, 5)}") for _ in range(n)]
    return gen


def question_stream(n: int, seed: int = 1) -> List[Tuple[str, dict]]:
    rng = random.Random(seed)
    topics = list(TOPIC_ACC)
    return [(f"q{i}", {"slice": (t := rng.choice(topics)), "truth": f"ans-{i}"}) for i in range(n)]
