"""Bind appearance-scored anchors to marked objects by their image-plane relation."""

import cv2
import numpy as np

from groundreach.grounding.markers import locate_marker
from groundreach.grounding.regions import iou


COLOR_RGB = {
    "red": (225, 45, 40), "orange": (235, 112, 28), "yellow": (230, 190, 30),
    "green": (40, 175, 55), "cyan": (28, 187, 222), "blue": (38, 79, 220),
    "purple": (146, 45, 205),
}


def hue_vector(rgb):
    vector = np.asarray(rgb, dtype=np.float64) - np.mean(rgb)
    return vector / max(np.linalg.norm(vector), 1e-6)


def anchor_score(image, candidate, phrase, use_shape):
    tint, geometry = phrase.split()
    height, width = image.shape[:2]
    x0, y0, x1, y1 = np.clip(candidate["box"], 0, [width, height, width, height])
    bgr = image[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
    if bgr.size == 0:
        return -np.inf
    foreground = (np.ptp(bgr, axis=2) > 32) & (bgr.max(2) > 45)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    if n <= 1:
        return -np.inf
    component = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    mask = labels == component
    observed = np.median(bgr[mask, ::-1], axis=0)
    color = float(hue_vector(observed) @ hue_vector(COLOR_RGB[tint]))
    if not use_shape:
        return color
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    width, height = cv2.minAreaRect(contour)[1]
    fill = cv2.contourArea(contour) / max(width * height, 1)
    expected = .99 if geometry == "box" else .79
    # A strong color mismatch should dominate shape and pretrained model tie-breaks.
    return color - .6 * abs(fill - expected) + .05 * candidate["semantic"] + .02 * candidate["score"]


def axis_center(box, axis):
    return (box[axis] + box[axis + 2]) / 2


def relation_box(pool, anchor, axis, direction, nearest, markers=None):
    if anchor is None:
        return None
    reference = axis_center(anchor, axis)
    options = []
    for index, candidate in enumerate(pool):
        box = candidate["box"]
        marker = locate_marker(markers, box) if markers is not None else None
        if markers is not None and marker is None:
            continue
        coordinate = marker[axis] if marker is not None else axis_center(box, axis)
        distance = direction * (coordinate - reference)
        if distance > 0 and iou(np.asarray(box), np.asarray(anchor)) < .5:
            options.append((distance if nearest else index, index, box))
    return min(options)[2] if options else None


def top_box(group):
    return group[0]["box"] if group else None
