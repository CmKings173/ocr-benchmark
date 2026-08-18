from ocr_benchmark.benchmark.runner import _classify_error
from ocr_benchmark.core.schemas import RunStatus
from ocr_benchmark.worker_entry import _error_status


def test_timeout_config_name_is_not_classified_as_request_timeout():
    error = TypeError("unexpected keyword argument 'timeout_seconds'")

    assert _classify_error(error) == RunStatus.EXCEPTION
    assert _error_status(error) == "EXCEPTION"


def test_real_timeout_is_classified_as_timeout():
    error = TimeoutError("worker request timed out after 300s")

    assert _classify_error(error) == RunStatus.TIMEOUT
    assert _error_status(error) == "TIMEOUT"
