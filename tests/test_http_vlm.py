import pytest

from ocr_benchmark.models.glm_ocr import GLMOCRAdapter


def test_remote_vlm_endpoint_requires_explicit_opt_in():
    adapter = GLMOCRAdapter(endpoint="https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="remote endpoint"):
        adapter.load()
