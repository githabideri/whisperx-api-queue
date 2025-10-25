import logging
import asyncio
import contextlib
import torch
import gc
from collections import defaultdict
from asyncio import Lock
from typing import Union, List, Optional, Tuple, Any

from whisperx import asr as whisperx_asr
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize

from dependencies import get_config

logger = logging.getLogger(__name__)

model_instances: dict[str, Any] = {}
model_locks: defaultdict[str, Lock] = defaultdict(Lock)

align_model_instances: dict[str, dict[str, Any]] = {}
alignment_locks: defaultdict[str, Lock] = defaultdict(Lock)

diarize_model_instances: dict[str, Any] = {}
diarization_locks: defaultdict[str, Lock] = defaultdict(Lock)

alignment_cache_mod_lock = Lock()


def unload_model_object(model_obj: Any):
    if model_obj is None:
        return
    with contextlib.suppress(Exception):
        model_obj.to("cpu")
    del model_obj
    gc.collect()
    torch.cuda.empty_cache()


class CustomWhisperModel(whisperx_asr.WhisperModel):
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: dict | None = None,
        **model_kwargs,
    ):
        super().__init__(
            model_size_or_path=model_size_or_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
            files=files,
            **model_kwargs,
        )
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.files = files
        self.model_kwargs = model_kwargs


def check_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.error("Could not determine device. Using 'cpu' instead.")
        return "cpu"


def _determine_inference_device():
    config = get_config()
    inference_device = config.whisper.inference_device.value
    if inference_device == "auto":
        inference_device = check_device()
    return inference_device


async def initialize_model(model_name: str) -> CustomWhisperModel:
    config = get_config()
    inference_device = _determine_inference_device()
    return CustomWhisperModel(
        model_size_or_path=model_name,
        device=inference_device,
        device_index=config.whisper.device_index,
        compute_type=config.whisper.compute_type.value,
        cpu_threads=config.whisper.cpu_threads,
        num_workers=config.whisper.num_workers,
        local_files_only=config.whisper.local_files_only,
        download_root=config.whisper.download_root,
    )


async def _get_or_init_model(
    key: str,
    cache_dict: dict,
    lock_dict: dict,
    init_func,
    log_reuse: str = "Reusing cached model instance for {key}",
    log_init: str = "Initializing model: {key}",
) -> Any:
    if key in cache_dict:
        logger.info(log_reuse.format(key=key))
        return cache_dict[key]

    async with lock_dict[key]:
        if key not in cache_dict:
            logger.info(log_init.format(key=key))
            cache_dict[key] = await init_func()
        return cache_dict[key]


async def load_model_instance(model_name: str):
    return await _get_or_init_model(
        key=model_name,
        cache_dict=model_instances,
        lock_dict=model_locks,
        init_func=lambda: initialize_model(model_name),
    )


async def _cleanup_alignment_cache_whitelist():
    config = get_config()
    whitelist = config.alignment.whitelist
    if not whitelist:
        return

    async with alignment_cache_mod_lock:
        for key in list(align_model_instances.keys()):
            if key not in whitelist:
                logger.info("Unloading alignment model for %s (not in whitelist).", key)
                align_model_data = align_model_instances.pop(key, None)
                if align_model_data is not None:
                    unload_model_object(align_model_data.get("model"))


async def load_align_model_cached(
    language_code: str,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    config = get_config()

    await _cleanup_alignment_cache_whitelist()

    inference_device = _determine_inference_device()

    selected_model_name = model_name
    if "multilingual" in config.alignment.models:
        selected_model_name = config.alignment.models["multilingual"]
        logger.info("Overriding with 'multilingual' alignment model: %s", selected_model_name)
    elif language_code in config.alignment.models:
        selected_model_name = config.alignment.models[language_code]
        logger.info("Using configured alignment model for '%s': %s", language_code, selected_model_name)

    if selected_model_name is not None and selected_model_name == config.alignment.models.get("multilingual"):
        cache_key = "multilingual"
    else:
        cache_key = language_code

    logger.debug("config.alignment.models = %s", config.alignment.models)
    logger.debug("Incoming language_code = %s, model_name param = %s", language_code, model_name)

    async def _init_alignment():
        try:
            loop = asyncio.get_running_loop()
            align_model, align_metadata = await loop.run_in_executor(
                None,
                lambda: whisperx_alignment.load_align_model(
                    language_code=language_code,
                    device=inference_device,
                    model_name=selected_model_name,
                    model_dir=model_dir,
                ),
            )
        except Exception as exc:
            logger.error("Failed to load alignment model for language '%s': %s", language_code, exc)
            raise

        return {"model": align_model, "metadata": align_metadata}

    model_data = await _get_or_init_model(
        key=cache_key,
        cache_dict=align_model_instances,
        lock_dict=alignment_locks,
        init_func=_init_alignment,
        log_reuse="Reusing cached alignment model for key: {key}",
        log_init="Initializing alignment model for key: {key}",
    )

    if not config.alignment.cache:
        async with alignment_cache_mod_lock:
            removed_data = align_model_instances.pop(cache_key, None)
            if removed_data is not None:
                logger.info("Unloading alignment model from cache (disabled): %s", cache_key)
                model_obj = removed_data.get("model")
                if model_obj is not None:
                    unload_model_object(model_obj)

    return model_data["model"], model_data["metadata"]


async def load_diarize_model_cached(model_name: str):
    config = get_config()
    inference_device = _determine_inference_device()

    async def _init_diarization():
        logger.info("Loading diarization pipeline for model: %s with device: %s", model_name, inference_device)
        return whisperx_diarize.DiarizationPipeline(model_name=model_name, device=inference_device)

    diarize_model = await _get_or_init_model(
        key=model_name,
        cache_dict=diarize_model_instances,
        lock_dict=diarization_locks,
        init_func=_init_diarization,
        log_reuse="Reusing cached diarization model for: {key}",
        log_init="Initializing diarization model: {key}",
    )

    if not config.diarization.cache:
        removed_model = diarize_model_instances.pop(model_name, None)
        if removed_model is not None:
            logger.info("Unloading diarization model from cache (disabled): %s", model_name)
            unload_model_object(removed_model)

    return diarize_model
