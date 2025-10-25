import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """Provide a TestClient with API key enforcement and stubbed queue writes."""
    monkeypatch.setenv("API_KEY", "test-key")

    from whisperx_api_queue import dependencies

    dependencies.get_config.cache_clear()
    import whisperx_api_queue.api as api_module

    api_module = importlib.reload(api_module)

    data_root = tmp_path / "jobs"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "DATA_ROOT", data_root)

    class DummyJob:
        def get_status(self) -> str:
            return "queued"

    monkeypatch.setattr(api_module, "_enqueue_job", lambda *args, **kwargs: DummyJob())

    client = TestClient(api_module.app)
    yield client
    client.close()

    dependencies.get_config.cache_clear()
    importlib.reload(api_module)


def _multipart():
    return {
        "files": {"file": ("clip.mp3", b"\x00\x01\x02", "audio/mpeg")},
        "data": {"model": "large-v3"},
    }


def test_bearer_header_allows_transcription(api_client):
    payload = _multipart()
    response = api_client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer test-key"},
        **payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "queued"
    assert body["job_id"]
    assert body["status_url"].endswith(body["job_id"])
    assert body["result_url"].endswith(body["job_id"])


def test_request_without_credentials_is_rejected(api_client):
    payload = _multipart()
    response = api_client.post("/v1/audio/transcriptions", **payload)
    assert response.status_code == 401


def test_x_api_key_header_is_accepted(api_client):
    payload = _multipart()
    response = api_client.post(
        "/v1/audio/transcriptions",
        headers={"X-API-Key": "test-key"},
        **payload,
    )
    assert response.status_code == 200
