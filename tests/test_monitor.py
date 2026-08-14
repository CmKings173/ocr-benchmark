from ocr_benchmark.benchmark.monitor import ResourceMonitor


def test_resource_monitor_returns_explicit_fields():
    sample = ResourceMonitor().sample()
    assert "rss_bytes" in sample
    assert "gpu_memory_bytes" in sample
    assert ResourceMonitor.summary([sample])["sample_count"] == 1
