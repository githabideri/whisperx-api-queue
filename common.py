import os
import redis
from rq import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
redis_conn = redis.from_url(REDIS_URL)
q = Queue("whisperx", connection=redis_conn)

def require_api_key(provided: str | None) -> None:
    expect = os.getenv("API_KEY", "")
    if expect:
        if not provided or provided != expect:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="invalid or missing API key")
