import pytest

from grounded.ingest import load_document


def test_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_document(path)


def test_plain_text_loads_verbatim(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("line one\nline two\n", encoding="utf-8")
    assert load_document(path) == "line one\nline two\n"


def test_rfc_page_furniture_is_stripped_but_lines_preserved(tmp_path):
    raw = (
        "Network Working Group\n"
        "Request for Comments: 9999\n"
        "\n"
        "1.  Introduction\n"
        "\n"
        "Real content here.\n"
        "\x0cBradner                 Standards Track                 [Page 2]\n"
        "RFC 9999          Some Title Here          March 1997\n"
        "More content after the page break.\n"
    )
    path = tmp_path / "rfc9999.txt"
    path.write_text(raw, encoding="utf-8")
    text = load_document(path)
    lines = text.splitlines()
    assert "[Page" not in text
    assert "\x0c" not in text
    # furniture lines blanked, not removed: line count is unchanged
    assert len(lines) == len(raw.splitlines())
    assert "More content after the page break." in lines[-1]


def test_pdf_text_extraction(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(_minimal_pdf(b"Grounded reads PDFs"))
    assert "Grounded reads PDFs" in load_document(path)


def _minimal_pdf(text: bytes) -> bytes:
    """Assemble a tiny single-page PDF with a correct xref table."""
    content = b"BT /F1 24 Tf 72 720 Td (" + text + b") Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    ).encode()
    return out
