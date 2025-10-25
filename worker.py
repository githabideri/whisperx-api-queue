import asyncio
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import redis

from config import MediaType, ResponseFormat
from dependencies import get_config
from formatters import format_transcription
from logger import setup_logger
from models import load_model_instance
from transcriber import transcribe_path

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
r = redis.from_url(REDIS_URL)

GPU_LOCK_KEY = os.getenv("GPU_LOCK_KEY", "whisperx:gpu:lock")
GPU_LOCK_TTL = int(os.getenv("GPU_LOCK_TTL", "600"))

CONFIG = get_config()
setup_logger(CONFIG.log_level)

ASYNC_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(ASYNC_LOOP)


def _run_async(coro):
    return ASYNC_LOOP.run_until_complete(coro)


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


def _write_artifact(path: Path, content: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _format_response_payload(
    transcription: dict[str, Any],
    response_format: ResponseFormat,
    highlight_words: bool,
) -> tuple[dict[str, Any], str, str]:
    response = format_transcription(transcription, response_format.value, highlight_words=highlight_words)
    body = response.body
    media_type = response.media_type or MediaType.APPLICATION_JSON.value

    if media_type == MediaType.APPLICATION_JSON.value:
        payload = json.loads(body.decode("utf-8"))
        serialized = json.dumps(payload, ensure_ascii=False)
    else:
        payload = {"text": body.decode("utf-8")}
        serialized = payload["text"]

    return payload, serialized, media_type


def _persist_primary_artifact(workdir: Path, response_format: ResponseFormat, serialized: str):
    extension_map = {
        ResponseFormat.JSON: "json",
        ResponseFormat.VERBOSE_JSON: "json",
        ResponseFormat.VTT_JSON: "json",
        ResponseFormat.TEXT: "txt",
        ResponseFormat.SRT: "srt",
        ResponseFormat.VTT: "vtt",
        ResponseFormat.AUD: "aud",
    }
    ext = extension_map.get(response_format, "json")
    path = workdir / f"result.{ext}"
    _write_artifact(path, serialized)
    return path


def _legacy_artifacts(
    workdir: Path,
    transcription: dict[str, Any],
    *,
    return_srt: bool,
    return_vtt: bool,
    return_tsv: bool,
    return_txt: bool,
):
    from whisperx.utils import WriteSRT, WriteTSV, WriteTXT, WriteVTT

    options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    segments = transcription.get("segments")
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
    aligned = {"language": transcription.get("language"), "segments": segments}
    base_dir = str(workdir)

    if return_srt:
        writer = WriteSRT(base_dir)
        path = workdir / "result.srt"
        writer({"language": aligned["language"], "segments": aligned["segments"]}, str(path), options)

    if return_vtt:
        writer = WriteVTT(base_dir)
        writer({"language": aligned["language"], "segments": aligned["segments"]}, str(workdir / "result.vtt"), options)

    if return_tsv:
        writer = WriteTSV(base_dir)
        writer({"language": aligned["language"], "segments": aligned["segments"]}, str(workdir / "result.tsv"), options)

    if return_txt:
        writer = WriteTXT(base_dir)
        writer({"language": aligned["language"], "segments": aligned["segments"]}, str(workdir / "result.txt"), options)


def _prepare_asr_options(
    *,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
    hotwords: str | None,
    suppress_numerals: bool,
) -> dict[str, Any]:
    return {
        "initial_prompt": prompt,
        "temperatures": temperature,
        "word_timestamps": word_timestamps,
        "hotwords": hotwords,
        "suppress_numerals": suppress_numerals,
    }


def _normalize_response_format(response_format: str | None) -> ResponseFormat:
    if response_format is None:
        return CONFIG.default_response_format
    try:
        return ResponseFormat(response_format)
    except ValueError as exc:
        raise ValueError(f"Unsupported response_format: {response_format}") from exc


def _run_transcription_pipeline(
    *,
    job_id: str,
    audio_path: str,
    original_filename: str,
    model: str,
    language: str | None,
    batch_size: int,
    chunk_size: int,
    asr_options: dict[str, Any],
    align: bool,
    diarize: bool,
    request_id: str,
    task: str,
) -> dict[str, Any]:
    model_instance = _run_async(load_model_instance(model))

    with gpu_lock():
        transcription = _run_async(
            transcribe_path(
                audio_path,
                original_filename=original_filename,
                batch_size=batch_size,
                chunk_size=chunk_size,
                asr_options=asr_options,
                language=language,
                whispermodel=model_instance,
                align=align,
                diarize=diarize,
                request_id=request_id,
                task=task,
            )
        )

    transcription["model"] = model
    return transcription


def run_openai_job(
    job_id: str,
    *,
    audio_path: str,
    original_filename: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
    temperature: float,
    timestamp_granularities: list[str] | None,
    hotwords: str | None,
    suppress_numerals: bool,
    highlight_words: bool,
    align: bool,
    diarize: bool,
    chunk_size: int,
    batch_size: int | None,
    task: str,
):
    from rq import get_current_job

    job = get_current_job()
    started = time.time()
    workdir = Path(audio_path).parent

    result_batch_size = batch_size or CONFIG.batch_size
    response_format_enum = _normalize_response_format(response_format)

    timestamp_granularities = timestamp_granularities or ["segment"]
    word_timestamps = "word" in timestamp_granularities

    model_name = model or CONFIG.whisper.model

    asr_options = _prepare_asr_options(
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
        hotwords=hotwords,
        suppress_numerals=suppress_numerals,
    )

    try:
        transcription = _run_transcription_pipeline(
            job_id=job_id,
            audio_path=audio_path,
            original_filename=original_filename,
            model=model_name,
            language=language,
            batch_size=result_batch_size,
            chunk_size=chunk_size,
            asr_options=asr_options,
            align=align,
            diarize=diarize,
            request_id=job_id,
            task=task,
        )

        payload_data, serialized_body, media_type = _format_response_payload(
            transcription, response_format_enum, highlight_words
        )

        primary_path = _persist_primary_artifact(workdir, response_format_enum, serialized_body)

        result = {
            "job_id": job_id,
            "task": task,
            "model": model_name,
            "language": transcription.get("language"),
            "response_format": response_format_enum.value,
            "media_type": media_type,
            "result_path": str(primary_path),
            "content": payload_data,
            "transcription": transcription,
            "duration_ms": int((time.time() - started) * 1000),
        }

        (workdir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result
    except Exception as exc:
        if job is not None:
            job.meta["error"] = str(exc)
            job.save_meta()
        raise


def run_whisperx_job(
    job_id: str,
    audio_path: str,
    language: str | None,
    diarize: bool,
    batch_size: int,
    return_srt: bool,
    return_vtt: bool,
    return_tsv: bool,
    return_txt: bool,
):
    result = run_openai_job(
        job_id,
        audio_path=audio_path,
        original_filename=os.path.basename(audio_path),
        model=CONFIG.whisper.model,
        language=language,
        prompt=None,
        response_format=ResponseFormat.JSON.value,
        temperature=0.0,
        timestamp_granularities=["segment"],
        hotwords=None,
        suppress_numerals=True,
        highlight_words=False,
        align=True,
        diarize=diarize,
        chunk_size=30,
        batch_size=batch_size,
        task="transcribe",
    )

    _legacy_artifacts(
        Path(audio_path).parent,
        result["transcription"],
        return_srt=return_srt,
        return_vtt=return_vtt,
        return_tsv=return_tsv,
        return_txt=return_txt,
    )

    return result
