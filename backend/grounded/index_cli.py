"""CLI: build and persist an index.

    python -m grounded.index_cli corpus/ --out data/index
"""

import argparse
import time
from pathlib import Path

from .config import get_settings
from .retrieval import Index
from .retrieval.embeddings import make_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Grounded index")
    parser.add_argument("corpus", type=Path, help="directory of .md/.txt/.pdf files")
    parser.add_argument("--out", type=Path, default=None, help="index output directory")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = args.out or Path(settings.index_dir)
    backend = make_backend(settings.embeddings, settings.st_model)

    started = time.time()
    index = Index.build(args.corpus, backend)
    index.save(out_dir)
    took = time.time() - started
    docs = {c.doc for c in index.chunks}
    print(f"indexed {len(index.chunks)} chunks from {len(docs)} documents in {took:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
