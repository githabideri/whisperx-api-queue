from whisperx.utils import WriteSRT, WriteVTT, WriteAudacity
from fastapi.responses import JSONResponse, Response

from config import MediaType


class ListWriter:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def get_output(self):
        return "".join(self.lines)

    def flush(self):
        pass


def update_options(kwargs, defaults):
    options = defaults.copy()
    options.update({key: kwargs.get(key, value) for key, value in defaults.items()})
    return options


def handle_whisperx_format(transcript, writer_class, options):
    writer = writer_class(output_dir=None)
    output = ListWriter()

    transcript["segments"]["language"] = transcript["language"]

    writer.write_result(transcript["segments"], output, options)

    return output.get_output()


def format_transcription(transcript, format, **kwargs) -> Response:
    defaults = {
        "max_line_width": 1000,
        "max_line_count": None,
        "highlight_words": kwargs.get("highlight_words", False),
    }
    options = update_options(kwargs, defaults)

    if format == "json":
        response_data = {"text": transcript.get("text", "")}
        return JSONResponse(content=response_data, media_type=MediaType.APPLICATION_JSON)
    if format == "verbose_json":
        return JSONResponse(content=transcript, media_type=MediaType.APPLICATION_JSON)
    if format == "vtt_json":
        transcript["vtt_text"] = handle_whisperx_format(transcript, WriteVTT, options)
        return JSONResponse(content=transcript, media_type=MediaType.APPLICATION_JSON)
    if format == "text":
        return Response(content=transcript.get("text", ""), media_type=MediaType.TEXT_PLAIN)
    if format == "srt":
        content = handle_whisperx_format(transcript, WriteSRT, options)
        return Response(content=content, media_type=MediaType.TEXT_PLAIN)
    if format == "vtt":
        content = handle_whisperx_format(transcript, WriteVTT, options)
        return Response(content=content, media_type=MediaType.TEXT_VTT)
    if format == "aud":
        content = handle_whisperx_format(transcript, WriteAudacity, options)
        return Response(content=content, media_type=MediaType.TEXT_PLAIN)
    raise ValueError(f"Unsupported format: {format}")
