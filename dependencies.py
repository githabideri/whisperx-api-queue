import json
import logging
import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Config


@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]

security = HTTPBearer(auto_error=False)

logger = logging.getLogger(__name__)


def load_api_keys(config: Config) -> tuple[set[str], dict[str, str]]:
    keys = set()
    mapping: dict[str, str] = {}

    env_key = os.getenv("API_KEY")
    if env_key:
        keys.add(env_key)

    if config.api_key:
        keys.add(config.api_key)

    if config.api_keys_file:
        try:
            with open(config.api_keys_file, "r", encoding="utf-8") as handle:
                file_keys = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API keys file error",
            ) from exc

        if isinstance(file_keys, dict):
            mapping = {str(key): str(value) for key, value in file_keys.items()}
            keys.update(mapping.keys())
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API keys file must be a JSON object mapping keys to client names",
            )

    return keys, mapping


def normalize_api_key_header(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return header.strip()


async def verify_api_key(
    config: ConfigDependency,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    valid_keys, key_mapping = load_api_keys(config)

    if not valid_keys:
        # No keys configured; allow anonymous access (mirrors server behavior)
        return

    provided_keys = set()

    if credentials is not None and credentials.credentials:
        provided_keys.add(credentials.credentials)

    normalized_header = normalize_api_key_header(x_api_key)
    if normalized_header:
        provided_keys.add(normalized_header)

    if not provided_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key")

    matched_key = next((key for key in provided_keys if key in valid_keys), None)
    if matched_key is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

    client_name = key_mapping.get(matched_key)
    if client_name:
        logger.info("Authorized request from client: '%s'", client_name)
    elif matched_key == config.api_key:
        logger.info("Authorized request using the default API key")
    elif matched_key == os.getenv("API_KEY"):
        logger.info("Authorized request using API_KEY environment variable")


ApiKeyDependency = Depends(verify_api_key)
