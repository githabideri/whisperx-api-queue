import os
from whisperx import transcribe as whisperx_transcribe
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
try:
    from whisperx import types as whisperx_types
    TranscriptionResult = whisperx_types.TranscriptionResult
except ImportError:
    from typing import Any

    TranscriptionResult = dict[str, Any]  # type: ignore[assignment]
from fastapi import UploadFile
import logging
import time
import tempfile
import torch
import gc
from typing import Any

from config import Language
from dependencies import get_config
from models import CustomWhisperModel, load_align_model_cached, load_diarize_model_cached

logger = logging.getLogger(__name__)

config = get_config()


def cleanup_cache_only():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


async def _transcribe_path(
    file_path: str,
    original_filename: str,
    *,
    batch_size: int,
    chunk_size: int,
    asr_options: dict[str, Any] | None,
    language: Language | str | None,
    whispermodel: CustomWhisperModel,
    align: bool,
    diarize: bool,
    request_id: str,
    task: str,
) -> TranscriptionResult:
    asr_options = asr_options or {}
    start_time = time.time()

    audio = None

    language_value = language.value if isinstance(language, Language) else language

    logger.info(
        "Request ID: %s - Transcribing %s with model: %s and options: %s",
        request_id,
        original_filename,
        whispermodel.model_size_or_path,
        asr_options,
    )

    try:
        model_loading_start = time.time()
        model = whisperx_transcribe.load_model(
            whisper_arch=whispermodel.model_size_or_path,
            device=whispermodel.device,
            compute_type=whispermodel.compute_type,
            language=language_value,
            asr_options=asr_options,
            vad_model=config.whisper.vad_model,
            vad_method=config.whisper.vad_method,
            vad_options=config.whisper.vad_options,
            model=whispermodel,
            task=task,
        )
        logger.info(
            "Request ID: %s - Loading model took %.2f seconds",
            request_id,
            time.time() - model_loading_start,
        )

        audio_loading_start = time.time()
        audio = whisperx_audio.load_audio(file_path)
        logger.info(
            "Request ID: %s - Loading audio took %.2f seconds",
            request_id,
            time.time() - audio_loading_start,
        )

        transcription_start = time.time()
        result = model.transcribe(
            audio=audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=config.whisper.num_workers,
            language=language_value,
            task=task,
        )
        logger.info(
            "Request ID: %s - Transcription took %.2f seconds",
            request_id,
            time.time() - transcription_start,
        )

        if align or diarize:
            alignment_model_start = time.time()
            logger.info("Request ID: %s - Loading alignment model", request_id)
            model_a, metadata = await load_align_model_cached(language_code=result["language"])
            logger.info("Request ID: %s - Alignment model loaded", request_id)
            logger.info(
                "Request ID: %s - Loading alignment model took %.2f seconds",
                request_id,
                time.time() - alignment_model_start,
            )

            alignment_start = time.time()
            result["segments"] = whisperx_alignment.align(
                transcript=result["segments"],
                model=model_a,
                align_model_metadata=metadata,
                audio=audio,
                device=whispermodel.device,
                return_char_alignments=False,
            )
            logger.info(
                "Request ID: %s - Alignment took %.2f seconds",
                request_id,
                time.time() - alignment_start,
            )

        if diarize:
            diarization_model_start = time.time()

            logger.info("Request ID: %s - Loading diarization model", request_id)

            diarize_model = await load_diarize_model_cached(model_name="tensorlake/speaker-diarization-3.1")

            logger.info(
                "Request ID: %s - Diarization model loaded. Loading took %.2f seconds. Starting diarization",
                request_id,
                time.time() - diarization_model_start,
            )

            diarize_start = time.time()

            diarize_segments = diarize_model(audio)

            result["segments"] = whisperx_diarize.assign_word_speakers(diarize_segments, result["segments"])

            logger.info(
                "Request ID: %s - Diarization took %.2f seconds",
                request_id,
                time.time() - diarize_start,
            )

        if align or diarize:
            result["text"] = "\n".join(
                [segment["text"].strip() for segment in result["segments"]["segments"] if segment["text"].strip()]
            )
        else:
            result["text"] = "\n".join([segment["text"].strip() for segment in result["segments"] if segment["text"].strip()])

        logger.info(
            "Request ID: %s - Transcription completed for %s in %.2f seconds",
            request_id,
            original_filename,
            time.time() - start_time,
        )
    except Exception as exc:
        logger.error(
            "Request ID: %s - Transcription failed for %s with error: %s",
            request_id,
            original_filename,
            exc,
        )
        raise
    finally:
        if config.audio_cleanup and audio is not None:
            del audio
            logger.debug("Request ID: %s - Audio data cleaned up", request_id)

        if config.cache_cleanup:
            cleanup_cache_only()
            logger.debug("Request ID: %s - Cache cleanup completed", request_id)

    return result


async def transcribe_upload_file(
    audio_file: UploadFile,
    *,
    batch_size: int,
    chunk_size: int,
    asr_options: dict[str, Any] | None,
    language: Language | str | None,
    whispermodel: CustomWhisperModel,
    align: bool,
    diarize: bool,
    request_id: str,
    task: str,
) -> TranscriptionResult:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    try:
        return await _transcribe_path(
            file_path=file_path,
            original_filename=audio_file.filename or os.path.basename(file_path),
            batch_size=batch_size,
            chunk_size=chunk_size,
            asr_options=asr_options,
            language=language,
            whispermodel=whispermodel,
            align=align,
            diarize=diarize,
            request_id=request_id,
            task=task,
        )
    finally:
        try:
            os.remove(file_path)
        except Exception:
            logger.error("Request ID: %s - Could not remove temporary file: %s", request_id, file_path)


async def transcribe_path(
    file_path: str,
    *,
    original_filename: str,
    batch_size: int,
    chunk_size: int,
    asr_options: dict[str, Any] | None,
    language: Language | str | None,
    whispermodel: CustomWhisperModel,
    align: bool,
    diarize: bool,
    request_id: str,
    task: str,
) -> TranscriptionResult:
    return await _transcribe_path(
        file_path=file_path,
        original_filename=original_filename,
        batch_size=batch_size,
        chunk_size=chunk_size,
        asr_options=asr_options,
        language=language,
        whispermodel=whispermodel,
        align=align,
        diarize=diarize,
        request_id=request_id,
        task=task,
    )
