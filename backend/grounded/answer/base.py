"""Answer providers behind one interface.

``extractive`` builds a quote-based answer directly from the retrieved
passages and needs no API key — it is the default so the whole demo runs
without any external account. The LLM providers (OpenAI, Anthropic, Ollama)
share the same contract: an answer whose sentences carry [n] citations into
the retrieval context.
"""

from dataclasses import dataclass, field

from ..retrieval import SearchHit


@dataclass
class AnswerSentence:
    text: str
    citations: list[int]  # context positions, 1-based (as shown to the user)


@dataclass
class Answer:
    provider: str
    text: str
    sentences: list[AnswerSentence] = field(default_factory=list)


class Provider:
    name = "base"

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:  # pragma: no cover
        raise NotImplementedError


def make_provider(name: str) -> Provider:
    if name == "extractive":
        from .extractive import ExtractiveProvider

        return ExtractiveProvider()
    if name in {"openai", "anthropic", "ollama"}:
        from .llm import LLMProvider

        return LLMProvider(name)
    raise ValueError(f"unknown provider: {name}")
