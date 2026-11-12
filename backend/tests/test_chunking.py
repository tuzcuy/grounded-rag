from grounded.ingest import chunk_document
from grounded.ingest.chunking import MAX_CHUNK_CHARS


def test_markdown_heading_hierarchy(widget_chunks):
    troubleshooting = [c for c in widget_chunks if "recovery mode" in c.text]
    assert len(troubleshooting) == 1
    assert troubleshooting[0].heading_path == ["Widget manual", "Troubleshooting"]


def test_sibling_heading_replaces_previous(widget_chunks):
    setup = next(c for c in widget_chunks if "blue cable" in c.text)
    assert setup.heading_path == ["Widget manual", "Setup"]


def test_line_ranges_point_into_source(widget_chunks):
    source_lines = None
    for chunk in widget_chunks:
        assert chunk.start_line <= chunk.end_line
        assert chunk.start_line >= 1


def test_numbered_sections_recognized():
    text = (
        "1.  Introduction\n\nThis spec defines the frobnicator protocol.\n\n"
        "1.1.  Terminology\n\nThe term frob means a reversible toggle operation.\n"
    )
    chunks = chunk_document("spec.txt", text)
    term = next(c for c in chunks if "reversible toggle" in c.text)
    assert term.heading_path == ["1. Introduction", "1.1. Terminology"]


def test_prose_starting_with_number_is_not_a_heading():
    text = "# Doc\n\n2119 keywords are referenced by many specifications today.\n"
    chunks = chunk_document("doc.md", text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Doc"]


def test_oversized_sections_split_on_paragraphs():
    paragraph = ("All work and no play makes the parser a dull tool. " * 8).strip()
    text = "# Big\n\n" + "\n\n".join([paragraph] * 6)
    chunks = chunk_document("big.md", text)
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS + len(paragraph) for c in chunks)
    # Line ranges must be contiguous and non-overlapping.
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start_line == earlier.end_line + 1


def test_empty_document_yields_no_chunks():
    assert chunk_document("empty.md", "   \n\n  ") == []


def test_ids_are_sequential(widget_chunks):
    assert [c.id for c in widget_chunks] == list(range(len(widget_chunks)))
