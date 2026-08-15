from ocr_benchmark.barcode.decoder import BarcodeResult, _decode_text_lists, _first_text, decode_exact_match


def test_barcode_exact_match_is_strict():
    result = BarcodeResult(engine="mock", image="x.png", status="SUCCESS", value="8938505971123")
    assert decode_exact_match(result, "8938505971123") is True
    assert decode_exact_match(result, "8938505971128") is False
    assert decode_exact_match(result, None) is None


def test_opencv_return_shapes_are_normalized():
    values, types = _decode_text_lists((True, ["123"], ["EAN-13"], None))
    assert values == ["123"]
    assert types == ["EAN-13"]
    assert _first_text(("", "QR-123", None)) == "QR-123"
