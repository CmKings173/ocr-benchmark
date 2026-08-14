from ocr_benchmark.benchmark.worker_manager import SubprocessWorker


def test_mock_worker_lifecycle():
    worker = SubprocessWorker()
    startup_ms = worker.start()
    try:
        assert startup_ms >= 0
        assert worker.request({"operation": "load"})["ok"] is True
        assert worker.request({"operation": "warmup", "iterations": 1})["ok"] is True
        response = worker.request({"operation": "predict", "image": "sample.png"})
        assert response["prediction"]["model"] == "mock"
    finally:
        worker.stop()
