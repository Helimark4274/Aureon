"""Adapter for a real LLM (Anthropic API). NOT exercised in sandbox tests (needs ANTHROPIC_API_KEY)."""
import os
from typing import List

from .types import Candidate


def anthropic_generator(model: str = "claude-sonnet-4-6", temperature: float = 1.0, max_tokens: int = 300):
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def gen(query: str, ctx: dict, n: int) -> List[Candidate]:
        system = ctx.get("system", "Answer concisely. Give only the final answer.")
        if ctx.get("sources"):
            system += "\nUse only these sources:\n" + "\n---\n".join(ctx["sources"])
        out = []
        for _ in range(n):
            r = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                       system=system, messages=[{"role": "user", "content": query}])
            out.append(Candidate("".join(b.text for b in r.content if b.type == "text")))
        return out
    return gen
