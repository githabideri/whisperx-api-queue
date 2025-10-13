import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from rq.job import Job

# Retry import compatible with multiple RQ versions
try:
    from rq.retry import Retry as RQRetry
except Exception:
    try:
        from rq import Retry as RQRetry
    except Exception:
        RQRetry = None

from common import q, redis_conn, require_api_key

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/srv/whisperx")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="WhisperX Queue API", version="0.1.1")

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
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    job_id = str(uuid.uuid4())
    workdir = DATA_ROOT / job_id
    workdir.mkdir(parents=True, exist_ok=False)

    dst = workdir / (file.filename or "input.wav")
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

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
    enqueue_kwargs = dict(
        kwargs=kwargs,
        job_id=job_id,
        ttl=24 * 3600,
        result_ttl=7 * 24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
    if RQRetry is not None:
        enqueue_kwargs["retry"] = RQRetry(max=3)

    job = q.enqueue("worker.run_whisperx_job", **enqueue_kwargs)
    return {"job_id": job_id, "state": job.get_status()}

@app.get("/status/{job_id}")
def status(job_id: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(404, "unknown job_id")
    return {
        "job_id": job_id,
        "state": job.get_status(),
        "error": getattr(job, "meta", {}).get("error"),
    }

@app.get("/result/{job_id}")
def result(job_id: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(404, "unknown job_id")
    if job.get_status() != "finished":
        raise HTTPException(409, "not ready")
    return JSONResponse(job.result)

@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    workdir = DATA_ROOT / job_id
    path = workdir / filename
    if not path.is_file():
        raise HTTPException(404, f"file not found: {filename}")
    return FileResponse(path)

@app.get("/healthz")
def healthz():
    return {"ok": True}
