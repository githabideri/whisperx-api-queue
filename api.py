import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import AfterValidator
from rq.job import Job
from starlette.middleware.base import BaseHTTPMiddleware

from common import q, redis_conn, require_api_key
from config import Config, Language, ResponseFormat
from dependencies import ApiKeyDependency, ConfigDependency, get_config
from logger import setup_logger

# Retry import compatible with multiple RQ versions
try:
    from rq.retry import Retry as RQRetry
except Exception:
    try:
        from rq import Retry as RQRetry
    except Exception:
        RQRetry = None


VERSION_FILE = Path(__file__).resolve().parent / "VERSION"
try:
    APP_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    APP_VERSION = "0.2.0"

CONFIG: Config = get_config()
setup_logger(CONFIG.log_level)
logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/srv/whisperx")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)


HAS_API_KEYS = bool(CONFIG.api_key or CONFIG.api_keys_file or os.getenv("API_KEY"))
OPENAI_DEPENDENCIES = [ApiKeyDependency] if HAS_API_KEYS else []


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def handle_default_openai_model(model_name: str) -> str:
    if model_name == "whisper-1":
        logger.info("Model 'whisper-1' requested; using default model '%s'", CONFIG.whisper.model)
        return CONFIG.whisper.model
    return model_name


ModelName = Annotated[str, AfterValidator(handle_default_openai_model)]


def _save_upload_file(file: UploadFile, workdir: Path) -> Path:
    filename = file.filename or "input.wav"
    dest = workdir / filename
    with dest.open("wb") as dst:
        shutil.copyfileobj(file.file, dst)
    return dest


def _enqueue_job(job_id: str, func: str, *, kwargs: dict) -> Job:
    enqueue_kwargs = dict(
        kwargs=kwargs,
        job_id=job_id,
        ttl=24 * 3600,
        result_ttl=7 * 24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
    if RQRetry is not None:
        enqueue_kwargs["retry"] = RQRetry(max=3)

    return q.enqueue(func, **enqueue_kwargs)


async def _get_timestamp_granularities(request: Request) -> list[Literal["segment", "word"]]:
    combinations = [
        [],
        ["segment"],
        ["word"],
        ["word", "segment"],
        ["segment", "word"],
    ]
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return ["segment"]
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    if timestamp_granularities not in combinations:
        raise HTTPException(
            status_code=422,
            detail=f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`.",
        )
    return timestamp_granularities


def _normalize_language(language: Language | None) -> str | None:
    if language is None:
        return CONFIG.default_language.value if CONFIG.default_language else None
    return language.value


def _normalize_response_format(response_format: ResponseFormat | None) -> ResponseFormat:
    return response_format or CONFIG.default_response_format


app = FastAPI(title="WhisperX Queue API", version=APP_VERSION)

if CONFIG.allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestIDMiddleware)


@app.post("/submit")
async def submit(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    diarize: bool = Form(False),
    batch_size: int = Form(16),
    return_srt: bool = Form(False),
    return_vtt: bool = Form(False),
    return_tsv: bool = Form(False),
    return_txt: bool = Form(False),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)

    job_id = str(uuid.uuid4())
    workdir = DATA_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=False)

    dst = _save_upload_file(file, workdir)

    kwargs = dict(
        job_id=job_id,
        audio_path=str(dst),
        language=language,
        diarize=diarize,
        batch_size=batch_size,
        return_srt=return_srt,
        return_vtt=return_vtt,
        return_tsv=return_tsv,
        return_txt=return_txt,
    )
    job = _enqueue_job(job_id, "worker.run_whisperx_job", kwargs=kwargs)
    return {"job_id": job_id, "state": job.get_status()}


@app.post(
    "/v1/audio/transcriptions",
    dependencies=OPENAI_DEPENDENCIES,
    description="Queue WhisperX transcription job (OpenAI-compatible surface).",
)
async def transcribe_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile = File(...),
    model: Annotated[ModelName | None, Form()] = None,
    language: Annotated[Language | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    suppress_numerals: Annotated[bool, Form()] = True,
    highlight_words: Annotated[bool, Form()] = False,
    align: Annotated[bool, Form()] = True,
    diarize: Annotated[bool, Form()] = False,
    chunk_size: Annotated[int, Form()] = 30,
    batch_size: Annotated[int | None, Form()] = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)

    timestamp_granularities = await _get_timestamp_granularities(request)

    if not align:
        if response_format in {ResponseFormat.SRT, ResponseFormat.VTT, ResponseFormat.AUD, ResponseFormat.VTT_JSON}:
            raise HTTPException(status_code=400, detail="Subtitles formats require alignment to be enabled.")
        if diarize:
            raise HTTPException(status_code=400, detail="Diarization requires alignment to be enabled.")

    job_id = str(uuid.uuid4())
    workdir = DATA_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=False)
    dst = _save_upload_file(file, workdir)

    response_format_enum = _normalize_response_format(response_format)
    language_value = _normalize_language(language)
    model_name = handle_default_openai_model(model or config.whisper.model)

    kwargs = dict(
        job_id=job_id,
        audio_path=str(dst),
        original_filename=file.filename or Path(dst).name,
        model=model_name,
        language=language_value,
        prompt=prompt,
        response_format=response_format_enum.value,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        hotwords=hotwords,
        suppress_numerals=suppress_numerals,
        highlight_words=highlight_words,
        align=align,
        diarize=diarize,
        chunk_size=chunk_size,
        batch_size=batch_size or config.batch_size,
        task="transcribe",
    )

    job = _enqueue_job(job_id, "worker.run_openai_job", kwargs=kwargs)

    return {
        "job_id": job_id,
        "state": job.get_status(),
        "status_url": f"/status/{job_id}",
        "result_url": f"/result/{job_id}",
    }


@app.post(
    "/v1/audio/translations",
    dependencies=OPENAI_DEPENDENCIES,
    description="Queue WhisperX translation job (OpenAI-compatible surface).",
)
async def translate_audio(
    config: ConfigDependency,
    request: Request,
    file: UploadFile = File(...),
    model: Annotated[ModelName | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = "",
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    chunk_size: Annotated[int, Form()] = 30,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)

    job_id = str(uuid.uuid4())
    workdir = DATA_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=False)
    dst = _save_upload_file(file, workdir)

    response_format_enum = _normalize_response_format(response_format)
    model_name = handle_default_openai_model(model or config.whisper.model)

    kwargs = dict(
        job_id=job_id,
        audio_path=str(dst),
        original_filename=file.filename or Path(dst).name,
        model=model_name,
        language=None,
        prompt=prompt,
        response_format=response_format_enum.value,
        temperature=temperature,
        timestamp_granularities=["segment"],
        hotwords=None,
        suppress_numerals=True,
        highlight_words=False,
        align=False,
        diarize=False,
        chunk_size=chunk_size,
        batch_size=config.batch_size,
        task="translate",
    )

    job = _enqueue_job(job_id, "worker.run_openai_job", kwargs=kwargs)

    return {
        "job_id": job_id,
        "state": job.get_status(),
        "status_url": f"/status/{job_id}",
        "result_url": f"/result/{job_id}",
    }


@app.get("/status/{job_id}")
def status(
    job_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as exc:
        raise HTTPException(404, "unknown job_id") from exc
    return {
        "job_id": job_id,
        "state": job.get_status(),
        "error": getattr(job, "meta", {}).get("error"),
    }


@app.get("/result/{job_id}")
def result(
    job_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as exc:
        raise HTTPException(404, "unknown job_id") from exc
    if job.get_status() != "finished":
        raise HTTPException(409, "not ready")
    return JSONResponse(job.result)


@app.get("/download/{job_id}/{filename}")
def download(
    job_id: str,
    filename: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    require_api_key(x_api_key, authorization)
    workdir = DATA_ROOT / job_id
    path = workdir / filename
    if not path.is_file():
        raise HTTPException(404, f"file not found: {filename}")
    return FileResponse(path)


@app.get("/healthz")
def healthz():
    return {"ok": True}
