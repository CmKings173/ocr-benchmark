from ocr_benchmark.barcode.decoder import BarcodeResult, decode_exact_match


def test_barcode_exact_match_is_strict():
    result = BarcodeResult(engine="mock", image="x.png", status="SUCCESS", value="8938505971123")
    assert decode_exact_match(result, "8938505971123") is True
    assert decode_exact_match(result, "8938505971128") is False
    assert decode_exact_match(result, None) is None
