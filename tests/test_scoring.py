from ocr_benchmark.reporting.scoring import aggregate_accuracy, gate_result, wilson_interval


def test_gate_and_aggregation():
    records = [{"prediction": {"status": "SUCCESS"}, "field_metrics": {"field_exact_match_accuracy": 1.0, "full_label_exact_match": True}, "cer": 0.0, "wer": 0.0}]
    summary = aggregate_accuracy(records)
    assert summary["full_label_accuracy"] == 1.0
    assert gate_result(summary, {"full_label_accuracy_min": 0.99})["eligible"] is True
    assert wilson_interval(1, 1)[0] < 1.0
