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


def test_company_named_price_columns_preserve_both_erp_tables():
    """Reproduce the ERP case: company names replace 'Supplier' in price columns."""
    source_files = (
        "Epoxy coated rebar comparison - reda project (1).xlsx; "
        "SAMCO quotation - Rebar Epoxy coated - MR 00967 - Reda Project (1).eml; "
        "Faisal Steel quotation - Rebar Epoxy coated - MR 00967 - Reda Project (1).eml"
    )
    material = (
        "| SN | Altered Material Name | Material description | Steel Pioneer Co Jubail Unit Price | "
        "Saudi Metal Coating Company Limited Unit Price | Rezayat Protective Coating Co. Ltd. Unit Price | "
        "Al Faisal Steel Products Co Dammam Unit Price | Source file(s) |\n"
        "|---:|---|---|---|---|---|---|---|\n"
        f"| 1 | 10MMX12METER | Epoxy coated rebar FBECR 10mm x 12meter ASTM A775 A615M,Gr.60; Unit: TON | 3785 SAR/TON | 4050 SAR/TON; 4050.00 SR/Mt | 4100 SAR/TON | 3780 SAR/TON; 3780 | {source_files} |\n"
        f"| 2 | 12MMX12METER | Epoxy coated rebar FBECR 12mm x 12meter ASTM A775 A615M,Gr.60; Unit: TON | 3435 SAR/TON | 3450 SAR/TON; 3450.00 SR/Mt | 3700 SAR/TON | 3275 SAR/TON; 3275 | {source_files} |\n"
        f"| 3 | 20MMX12METER | Epoxy coated rebar FBECR 20mm x 12meter ASTM A775 A615M,Gr.60; Unit: TON | 3235 SAR/TON | 3115 SAR/TON; 3115.00 SR/Mt | 3500 SAR/TON | 3075 SAR/TON; 3075 | {source_files} |\n"
        f"| 4 | 25MMX12METER | Epoxy coated rebar FBECR 25mm x 12meter ASTM A775 A615M,Gr.60; Unit: TON | 3235 SAR/TON | 3115 SAR/TON; 3115.00 SR/Mt | 3500 SAR/TON | 3075 SAR/TON; 3075 | {source_files} |"
    )
    payment = (
        "| Supplier Name | Payment Terms | Quotation Validity |\n|---|---|---|\n"
        "| Steel Pioneer Co Jubail | 100% Advance | |\n"
        "| Saudi Metal Coating Company Limited | 100% Advance; 100% ADVANCE PAYMENT ALONG WITH PURCHASE ORDER. | |\n"
        "| Rezayat Protective Coating Co. Ltd. | 100% Advance | |\n"
        "| Al Faisal Steel Products Co Dammam | 100% Advance | |"
    )
    tables = [markdown_table_to_df(block).fillna("").to_dict(orient="records")
              for block in (material + "\n\n" + payment).split("\n\n")]
    assert [len(table) for table in tables] == [4, 4]
    assert tables[0][0]["Steel Pioneer Co Jubail Unit Price"] == "3785 SAR/TON"
    assert tables[0][0]["Saudi Metal Coating Company Limited Unit Price"] == "4050 SAR/TON; 4050.00 SR/Mt"
    assert tables[0][3]["Al Faisal Steel Products Co Dammam Unit Price"] == "3075 SAR/TON; 3075"
    assert tables[0][0]["Source file(s)"] == source_files
    assert all(row["Quotation Validity"] == "" for row in tables[1])


def test_txt_extraction_supports_utf8():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "quote.txt"
        path.write_text("Supplier: Test\nRice | SAR 10", encoding="utf-8")
        assert "Supplier: Test" in extract_text_from_file(str(path))
