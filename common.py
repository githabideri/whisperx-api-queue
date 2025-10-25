import os
import redis
from rq import Queue

from fastapi import HTTPException

from dependencies import get_config, load_api_keys, normalize_api_key_header

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
redis_conn = redis.from_url(REDIS_URL)
q = Queue("whisperx", connection=redis_conn)


def require_api_key(x_api_key: str | None = None, authorization: str | None = None) -> None:
    config = get_config()
    valid_keys, _ = load_api_keys(config)
    if not valid_keys:
        return

    provided_keys = set()

    normalized_header = normalize_api_key_header(authorization)
    if normalized_header:
        provided_keys.add(normalized_header)

    normalized_x_api_key = normalize_api_key_header(x_api_key)
    if normalized_x_api_key:
        provided_keys.add(normalized_x_api_key)

    if not provided_keys:
        raise HTTPException(status_code=401, detail="invalid or missing API key")

    if provided_keys.isdisjoint(valid_keys):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
