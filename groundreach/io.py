"""Read and write archived JSON and verify image provenance."""

import hashlib
import json

from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def image(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Image hash changed: {path}")
    with Image.open(path) as source:
        return source.convert("RGB")
