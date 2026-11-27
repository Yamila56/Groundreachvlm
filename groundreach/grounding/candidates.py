"""Load candidate groups and verify their image inputs."""

import cv2

from groundreach.io import read, sha256_file


def checked_image(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Input hash changed: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def candidate_index(path):
    source = read(path)
    entries = {(r["image_sha256"], r["phrase"]): r["boxes"] for r in source}
    if len(entries) != len(source):
        raise ValueError(f"Duplicate candidate groups in {path}")
    return entries
