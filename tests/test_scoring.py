from ocr_benchmark.reporting.scoring import aggregate_accuracy, gate_result, wilson_interval


def test_gate_and_aggregation():
    records = [{"prediction": {"status": "SUCCESS"}, "field_metrics": {"field_exact_match_accuracy": 1.0, "full_label_exact_match": True}, "cer": 0.0, "wer": 0.0}]
    summary = aggregate_accuracy(records)
    assert summary["full_label_accuracy"] == 1.0
    assert gate_result(summary, {"full_label_accuracy_min": 0.99})["eligible"] is True
    assert wilson_interval(1, 1)[0] < 1.0


def test_gate_rejects_run_without_successful_predictions():
    summary = aggregate_accuracy([{"prediction": {"status": "EXCEPTION"}}])
    result = gate_result(summary, {})
    assert result["valid"] is False
    assert result["eligible"] is False


def test_missing_text_annotation_is_not_scored_as_zero():
    summary = aggregate_accuracy(
        [{"prediction": {"status": "SUCCESS"}, "field_metrics": {"field_exact_match_accuracy": 1.0, "full_label_exact_match": True}}]
    )
    assert summary["mean_cer"] is None
    assert summary["mean_wer"] is None


def test_zero_failure_rate_gets_full_reliability():
    from ocr_benchmark.reporting.scoring import composite_score

    score = composite_score(
        {
            "eligible": True,
            "full_label_accuracy": 1.0,
            "critical_field_accuracy": 1.0,
            "failure_rate": 0.0,
            "p95_ms": 100.0,
        }
    )
    assert score is not None
    assert score > 0.9
