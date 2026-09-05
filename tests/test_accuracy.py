import tempfile
from pathlib import Path

from summarize_doc import canonical_key, extract_text_from_file, markdown_table_to_df


def test_canonical_key_keeps_dimensions_distinct():
    assert canonical_key("WOOD PANEL 4 * 8") != canonical_key("WOOD PANEL 4 * 10")
    assert canonical_key("150 MM - 12 MM") == "150MMX12MM"


def test_markdown_table_parser_preserves_empty_cells():
    table = "| Item | Supplier A | Supplier B |\n|---|---|---|\n| Rice | SAR 10 |  |"
    result = markdown_table_to_df(table)
    assert result.iloc[0]["Supplier B"] == ""


def test_txt_extraction_supports_utf8():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "quote.txt"
        path.write_text("Supplier: Test\nRice | SAR 10", encoding="utf-8")
        assert "Supplier: Test" in extract_text_from_file(str(path))
