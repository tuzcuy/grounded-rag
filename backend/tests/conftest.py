import os
from pathlib import Path

import pytest

# Keyless + offline: every test runs on the deterministic hashing embedder.
os.environ["GROUNDED_EMBEDDINGS"] = "hashing"

from grounded.ingest import chunk_document  # noqa: E402
from grounded.retrieval import Index  # noqa: E402
from grounded.retrieval.embeddings import HashingEmbedder  # noqa: E402

DOC_A = """# Widget manual

## Setup

To install the widget, connect the blue cable to the power port and wait
for the indicator to turn green. The widget requires a 12 volt supply.

## Troubleshooting

If the indicator blinks red, the firmware update failed and the widget
must be restarted in recovery mode by holding the reset button.
"""

DOC_B = """# Gadget guide

## Battery

The gadget battery lasts for twelve hours of continuous recording and
charges fully in ninety minutes using the bundled charger.

## Warranty

Warranty claims require the original purchase receipt and are handled
within fourteen business days by the service center.
"""


@pytest.fixture(scope="session")
def backend():
    return HashingEmbedder(dim=128)


@pytest.fixture(scope="session")
def tiny_index(backend, tmp_path_factory):
    corpus = tmp_path_factory.mktemp("corpus")
    (corpus / "widgets.md").write_text(DOC_A, encoding="utf-8")
    (corpus / "gadgets.md").write_text(DOC_B, encoding="utf-8")
    return Index.build(Path(corpus), backend)


@pytest.fixture()
def widget_chunks():
    return chunk_document("widgets.md", DOC_A)
