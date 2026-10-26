import whisper
from pathlib import Path
import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

_model = None


def _ensure_ffmpeg():
    """Whisper shells out to `ffmpeg`. If it isn't on PATH, expose the binary bundled with imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg
    bin_dir = Path(tempfile.gettempdir()) / "civicfix_ffmpeg"
    target = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_path: Path) -> dict:
    """
    Returns:
        {
            "success": bool,
            "transcript": str | None,
            "language": str | None,
            "error": str | None
        }
    """
    try:
        _ensure_ffmpeg()
        model = _get_model()
        result = model.transcribe(str(audio_path), fp16=False)
        return {
            "success": True,
            "transcript": result["text"].strip(),
            "language": result.get("language"),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            "success": False,
            "transcript": None,
            "language": None,
            "error": str(e),
        }
