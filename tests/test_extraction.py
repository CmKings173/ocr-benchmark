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
