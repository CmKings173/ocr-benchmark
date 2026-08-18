from ocr_benchmark.models.glm_output import parse_glm_content


def test_glm_nested_json_is_flattened_without_value_normalization():
    raw_text, fields = parse_glm_content(
        """```json
{
  "ACME INDUSTRIAL BEARING UNIT": {
    "SKU": "MX-P2NC0JT",
    "LOT": "L20260801-B90",
    "QTY": "1000 BOX",
    "S/N": "SN-HJBSGT25ZN",
    "PO": "PO-202608-0001"
  },
  "CODE": "8914852027678"
}
```"""
    )

    assert raw_text.startswith("```json")
    assert fields == {
        "sku": "MX-P2NC0JT",
        "lot": "L20260801-B90",
        "quantity": "1000",
        "unit": "BOX",
        "serial": "SN-HJBSGT25ZN",
        "po_number": "PO-202608-0001",
        "barcode": "8914852027678",
    }


def test_glm_text_object_is_parsed():
    raw_text, fields = parse_glm_content(
        {
            "text": "NOVA LOGISTICS\nSKU: MX-I23\nLOT: L-1\nQTY: 10 PCS\nS/N: SN-1\nPO: PO-1\nCODE: 123"
        }
    )

    assert raw_text.startswith("NOVA LOGISTICS")
    assert fields["sku"] == "MX-I23"
    assert fields["quantity"] == "10"
    assert fields["unit"] == "PCS"
    assert fields["barcode"] == "123"


def test_glm_malformed_json_like_lines_are_parsed():
    _, fields = parse_glm_content(
        '"SKU": "ABC-I23",\n"LOT": "LOT-9",\n"QTY": "2 PCS",\n'
        '"S/N": "SN-9",\n"PO": "PO-9",\n"CODE": "000123"'
    )

    assert fields == {
        "sku": "ABC-I23",
        "lot": "LOT-9",
        "quantity": "2",
        "unit": "PCS",
        "serial": "SN-9",
        "po_number": "PO-9",
        "barcode": "000123",
    }


def test_glm_named_field_list_is_parsed():
    _, fields = parse_glm_content(
        {
            "raw_text": "SKU: ABC",
            "fields": [{"name": "SKU", "value": "ABC"}, {"name": "QTY", "value": "3 PCS"}],
        }
    )

    assert fields["sku"] == "ABC"
    assert fields["quantity"] == "3"
    assert fields["unit"] == "PCS"
