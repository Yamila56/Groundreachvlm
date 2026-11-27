"""Read and write archived JSON and verify image provenance."""

import hashlib
import json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def image(path, expected):
    from PIL import Image

    if sha256_file(path) != expected:
        raise ValueError(f"Image hash changed: {path}")
    with Image.open(path) as source:
        return source.convert("RGB")


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model_weights(directory, expected):
    """Verify the unsharded Safetensors file used by the recorded Qwen3-VL protocol."""
    weights = directory / "model.safetensors"
    if sha256_file(weights) != expected:
        raise ValueError(f"SHA-256 mismatch for {weights}")
