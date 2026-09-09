from structured_quotations import to_markdown


def test_structured_result_keeps_legacy_two_table_format():
    markdown = to_markdown({
        "suppliers": [{"name": "Supplier A", "payment_terms": "Advance", "quotation_validity": "7 days"}],
        "items": [{"name": "10MM", "description": "Rebar; Unit: TON", "source_files": ["quote.pdf"],
                   "prices": [{"supplier": "Supplier A", "value": "100 SAR/TON"}]}],
    })
    assert "Supplier A Unit Price" in markdown
    assert "100 SAR/TON" in markdown
    assert "| Supplier Name | Payment Terms | Quotation Validity |" in markdown
    assert "| Supplier A | Advance | 7 days |" in markdown
