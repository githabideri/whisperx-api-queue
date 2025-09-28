Welcome to this small project of mine, I wanted a small API server that specifically works with WhisperX and has the ability to handle several requests at the same time (by letting them wait).
Will be improved over time, don't expect miracles, I am not a programmer. This project was mainly written by GPT-5 (in its different variants) and only composed/conducted/slopped together by me.
Might have security issues, so don't expose over the internet and always be careful what you download over the internet.

# whisperx-api-queue (v0.1.1)

Queue-based HTTP API for [WhisperX](https://github.com/m-bain/whisperX) with internal GPU exclusivity via a Redis lock.
Built for single-GPU hosts (LXC ok) with optional Tailscale exposure.

## Features
- FastAPI HTTP surface: `/submit`, `/status/{job_id}`, `/result/{job_id}`, `/healthz`
- RQ/Redis job queue (no request timeouts on long media)
- Single-GPU exclusivity (Redis lock) so only one WhisperX job runs at a time
- Persistent artifacts per job (`/srv/whisperx/<job_id>/result.json`, `result.srt`)
- Minimal deps; WhisperX + Torch are assumed installed in your venv
- Model configurable via env (`WHISPER_MODEL`, default **large-v3**)

## Requirements
- Ubuntu 24.04 LXC (or any Linux host)
- Redis server (apt `redis-server`)
- Python 3.10+ virtualenv with: torch/torchaudio (CUDA), whisperx
- This repo deps: `fastapi`, `uvicorn[standard]`, `redis`, `rq`, `python-multipart`

## Quickstart (manual run)
```bash
# inside your venv
pip install -r requirements.txt

# terminal 1: start worker
export PYTHONPATH=$(pwd)
export WHISPER_MODEL=large-v3
export GPU_LOCK_TTL=600
rq worker whisperx

# terminal 2: start API on LAN
uvicorn api:app --host 0.0.0.0 --port 7860 --workers 1

# terminal 3 (client): submit, poll, fetch
JOB_ID=$(curl -s -X POST "http://<HOST>:7860/submit"   -F file=@/path/to/audio.mp3   -F language=de -F diarize=false -F return_srt=true -F batch_size=8 | jq -r .job_id)

curl -s "http://<HOST>:7860/status/$JOB_ID" | jq
curl -s "http://<HOST>:7860/result/$JOB_ID" -o result.json
jq '.model, .segments[0:3] | map({start,end,text})' result.json
jq -r '.srt' result.json | sed -n '1,40p'
```

## Systemd deployment
1. **Install deps (root once)**
```bash
apt update && apt install -y redis-server
mkdir -p /srv/whisperx
chown -R <USER>:<USER> /srv/whisperx
```

2. **Env file** (root): `/etc/default/whisperx-api-queue`
```
API_KEY=change-me-please
DATA_ROOT=/srv/whisperx
REDIS_URL=redis://127.0.0.1:6379/0
PYTHONPATH=/home/<USER>/whisperx-api-queue
WHISPER_MODEL=large-v3
COMPUTE_TYPE=float16
CUDA_VISIBLE_DEVICES=0
GPU_LOCK_TTL=600
# HF_TOKEN=    # only if using diarization
```

3. **Units** (root):
- `/etc/systemd/system/whisperx-api-queue.service` → see `deploy/systemd/whisperx-api-queue.service`
- `/etc/systemd/system/whisperx-worker-queue.service` → see `deploy/systemd/whisperx-worker-queue.service`

4. **Enable & start**
```bash
systemctl daemon-reload
systemctl enable redis-server whisperx-worker-queue whisperx-api-queue
systemctl start whisperx-worker-queue whisperx-api-queue
journalctl -u whisperx-api-queue -f
journalctl -u whisperx-worker-queue -f
```

## Endpoints
- `POST /submit` → `{ job_id, state }`
  - form fields: `file`, `language?`, `diarize?`, `batch_size?`, `return_srt?`
- `GET /status/{job_id}` → `{ job_id, state, error? }`
- `GET /result/{job_id}` → `{ segments[], language, model, diarized, srt? }`
- `GET /healthz` → `{ ok: true }`

## Notes
- **Diarization**: Requires HuggingFace token: set `HF_TOKEN` and accept pyannote model licenses.
- **GPU lock**: Only serializes **this** service; other GPU consumers (e.g., Ollama in another LXC) are not coordinated.
  - Clear stuck lock: `redis-cli DEL whisperx:gpu:lock`
- **OOM**: Use smaller `batch_size` or switch to `medium` model.
- **Tailscale**: For tailnet-only exposure, bind API to `127.0.0.1` and run `tailscale serve http / http://127.0.0.1:7860`.

## License
MIT
