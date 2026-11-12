"""LLM-backed providers (OpenAI, Anthropic, Ollama) sharing one prompt
contract: answer ONLY from the numbered context and cite every sentence
with [n] markers. Requires the relevant API key (or a local Ollama)."""

import json
import os

import httpx

from ..retrieval import SearchHit
from .base import Answer, AnswerSentence, Provider
from .citations import resolve_citations

SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages. "
    "Cite every sentence with the matching [n] markers. "
    "If the context does not contain the answer, say so explicitly."
)


def _context_block(hits: list[SearchHit]) -> str:
    parts = []
    for pos, hit in enumerate(hits, start=1):
        head = " > ".join(hit.chunk.heading_path) or hit.chunk.doc
        parts.append(f"[{pos}] ({hit.chunk.doc} — {head})\n{hit.chunk.text}")
    return "\n\n".join(parts)


class LLMProvider(Provider):
    def __init__(self, name: str):
        self.name = name

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        prompt = (
            f"Context passages:\n\n{_context_block(hits)}\n\n"
            f"Question: {question}\n\nAnswer with [n] citations:"
        )
        raw = self._complete(prompt)
        sentences = [
            AnswerSentence(text=t, citations=c) for t, c in resolve_citations(raw, len(hits))
        ]
        return Answer(provider=self.name, text=raw, sentences=sentences)

    def _complete(self, prompt: str) -> str:
        if self.name == "openai":
            return self._openai(prompt)
        if self.name == "anthropic":
            return self._anthropic(prompt)
        return self._ollama(prompt)

    def _openai(self, prompt: str) -> str:
        key = _require_env("OPENAI_API_KEY")
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        key = _require_env("ANTHROPIC_API_KEY")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def _ollama(self, prompt: str) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        response = httpx.post(
            f"{host}/api/generate",
            json={
                "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return json.loads(response.text)["response"]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — use provider=extractive for keyless mode")
    return value
