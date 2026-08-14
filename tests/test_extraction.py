from ocr_benchmark.extraction.fields import extract_fields
from ocr_benchmark.models.helpers import normalize_bbox


def test_extraction_parses_structure_without_value_repair():
    fields = extract_fields("SKU: ABC-I23\nQTY: 500", {}, {"sku": "ABC-123", "qty": 500})
    assert fields == {"sku": "ABC-I23", "qty": 500}


def test_normalize_polygon_bbox():
    assert normalize_bbox([[10, 20], [30, 20], [30, 40], [10, 40]]) == [10.0, 20.0, 30.0, 40.0]
