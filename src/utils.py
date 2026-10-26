import base64
from pathlib import Path
from PIL import Image
import io


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def save_temp_audio(uploaded_file) -> Path:
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return Path(f.name)


def load_image_pil(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.read()))
