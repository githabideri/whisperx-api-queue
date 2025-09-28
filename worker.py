import os, json, time
from pathlib import Path
from contextlib import contextmanager

import torch
import whisperx
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
r = redis.from_url(REDIS_URL)

# -------- SRT helper --------
def _fmt_ts(sec: float) -> str:
    if sec is None:
        sec = 0.0
    if sec < 0:
        sec = 0.0
    ms_total = int(round(sec * 1000))
    h = ms_total // 3_600_000
    m = (ms_total % 3_600_000) // 60_000
    s = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments, path: str):
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_ts(seg.get("start"))
        end = _fmt_ts(seg.get("end"))
        text = (seg.get("text") or "").strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text if text else "")
        lines.append("")  # blank line
    Path(path).write_text("\n".join(lines), encoding="utf-8")

# -------- GPU exclusivity lock --------
GPU_LOCK_KEY = os.getenv("GPU_LOCK_KEY", "whisperx:gpu:lock")
GPU_LOCK_TTL = int(os.getenv("GPU_LOCK_TTL", str(600)))  # default 10 min

@contextmanager
def gpu_lock():
    lock = r.lock(GPU_LOCK_KEY, timeout=GPU_LOCK_TTL, blocking_timeout=60)
    if not lock.acquire(blocking=True):
        raise RuntimeError("could not acquire GPU lock")
    try:
        yield
    finally:
        try:
            lock.release()
        except Exception:
            pass

_model = None
_align_model = None
_align_meta = None
_diarizer = None

def _lazy_models(language: str | None, diarize: bool):
    global _model, _align_model, _align_meta, _diarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = os.getenv("COMPUTE_TYPE", "float16")
    model_name = os.getenv("WHISPER_MODEL", "large-v3")  # default v3

    if _model is None:
        _model = whisperx.load_model(model_name, device, compute_type=compute_type)
    if _align_model is None:
        _align_model, _align_meta = whisperx.load_align_model(language_code=language, device=device)
    if diarize and _diarizer is None:
        hf = os.getenv("HF_TOKEN")
        if not hf:
            raise RuntimeError("HF_TOKEN missing for diarization")
        _diarizer = whisperx.DiarizationPipeline(device=device, use_auth_token=hf)
    return _model, _align_model, _align_meta, _diarizer

def run_whisperx_job(job_id: str, audio_path: str, language: str | None,
                     diarize: bool, batch_size: int, return_srt: bool):
    from rq import get_current_job
    job = get_current_job()
    started = time.time()
    workdir = Path(audio_path).parent

    try:
        # Load models & audio OUTSIDE the lock
        model, align_model, align_meta, diarizer = _lazy_models(language, diarize)
        audio = whisperx.load_audio(audio_path)

        # GPU-bound steps INSIDE the lock
        with gpu_lock():
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            aligned = whisperx.align(result["segments"], align_model, align_meta, audio, device)

            if diarize:
                dz = diarizer(audio)
                aligned = whisperx.assign_word_speakers(dz, aligned)

        payload = {
            "job_id": job_id,
            "language": aligned.get("language"),
            "segments": aligned["segments"],
            "diarized": bool(diarize),
            "model": os.getenv("WHISPER_MODEL", "large-v3"),
        }

        if return_srt:
            srt_path = workdir / "result.srt"
            write_srt(aligned["segments"], str(srt_path))
            payload["srt"] = srt_path.read_text(encoding="utf-8")

        (workdir / "result.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        payload["duration_ms"] = int((time.time() - started) * 1000)
        return payload

    except Exception as e:
        if job is not None:
            job.meta["error"] = str(e)
            job.save_meta()
        raise
