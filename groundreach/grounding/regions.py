"""Region geometry shared by the grounding and crop studies."""

import numpy as np


def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(1, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - intersection))


def rectangle(box, shape, scale=1):
    h, w = shape[:2]
    center = (box[:2]+box[2:])/2
    delta = (box[2:]-box[:2])*scale/2
    ends = np.clip([*np.floor(center-delta), *np.ceil(center+delta)], 0, [w, h, w, h]).astype(int)
    return ends.tolist() if np.all(ends[2:] > ends[:2]) else None


def region(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]
