"""Runtime configuration, resolved from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("GROUNDED_INDEX_DIR", "data/index"))
    )
    # "st" = sentence-transformers (real model), "hashing" = deterministic offline embedder
    embeddings: str = field(default_factory=lambda: os.environ.get("GROUNDED_EMBEDDINGS", "st"))
    st_model: str = field(
        default_factory=lambda: os.environ.get(
            "GROUNDED_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Default answer provider; "extractive" needs no API key.
    provider: str = field(default_factory=lambda: os.environ.get("GROUNDED_PROVIDER", "extractive"))
    top_k: int = field(default_factory=lambda: int(os.environ.get("GROUNDED_TOP_K", "8")))


def get_settings() -> Settings:
    return Settings()
