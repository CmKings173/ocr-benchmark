from ocr_benchmark.extraction.fields import extract_fields
from ocr_benchmark.models.helpers import normalize_bbox, to_plain


def test_extraction_parses_structure_without_value_repair():
    fields = extract_fields("SKU: ABC-I23\nQTY: 500", {}, {"sku": "ABC-123", "qty": 500})
    assert fields == {"sku": "ABC-I23", "qty": 500}


def test_normalize_polygon_bbox():
    assert normalize_bbox([[10, 20], [30, 20], [30, 40], [10, 40]]) == [10.0, 20.0, 30.0, 40.0]


def test_to_plain_handles_vendor_objects_without_dict():
    class VendorScalar:
        __slots__ = ()

        def __str__(self):
            return "vendor-value"

    assert to_plain(VendorScalar()) == "vendor-value"


def test_to_plain_handles_array_like_values():
    class ArrayLike:
        def tolist(self):
            return [[1, 2], [3, 4]]

    assert to_plain(ArrayLike()) == [[1, 2], [3, 4]]


def test_extract_fields_supports_multiline_label_format():
    raw_text = """SKU:
MX-HD3CWYL
LOT:
L20260823-D18
QTY:
10 PCS
S/N:
SN-RV0ATNNPAZ
PO:
PO-202608-0017
CODE: 8999519665641"""
    expected = {
        "sku": "MX-HD3CWYL",
        "lot": "L20260823-D18",
        "quantity": "10",
        "unit": "PCS",
        "serial": "SN-RV0ATNNPAZ",
        "po_number": "PO-202608-0017",
        "barcode": "8999519665641",
    }
    assert extract_fields(raw_text, {}, expected) == expected


def test_empty_adapter_fields_do_not_block_raw_text_fallback():
    expected = {"sku": "MX-P2NC0JT", "lot": "L20260801-B90"}
    assert extract_fields(
        "SKU:\nMX-P2NC0JT\nLOT:\nL20260801-B90",
        {"sku": "", "lot": ""},
        expected,
    ) == expected


def test_structured_quantity_value_is_split_from_unit():
    expected = {"quantity": "1000", "unit": "BOX"}
    assert extract_fields("", {"QTY": "1000 BOX"}, expected) == expected
