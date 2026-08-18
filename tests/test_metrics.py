from ocr_benchmark.metrics.fields import field_exact_metrics, structural_normalize_fields
from ocr_benchmark.metrics.text import cer, wer


def test_structural_normalization_does_not_change_values():
    fields = structural_normalize_fields({"SKU": "ABC-I23"}, {"SKU": "sku"})
    assert fields == {"sku": "ABC-I23"}


def test_strict_field_metrics_reject_confusion():
    result = field_exact_metrics({"sku": "ABC-123"}, {"sku": "ABC-I23"}, ["sku"])
    assert result["field_exact_match_accuracy"] == 0.0
    assert result["full_label_exact_match"] is False


def test_critical_field_metrics_are_separate():
    result = field_exact_metrics({"sku": "ABC", "unit": "PCS"}, {"sku": "WRONG", "unit": "PCS"}, ["sku", "unit"], ["sku"])
    assert result["field_exact_match_accuracy"] == 0.5
    assert result["critical_field_exact_match_accuracy"] == 0.0


def test_text_metrics():
    assert cer("ABC", "ABD") == 1 / 3
    assert wer("A B", "A C") == 0.5
    assert cer("", "anything") is None
    assert wer("", "anything") is None
