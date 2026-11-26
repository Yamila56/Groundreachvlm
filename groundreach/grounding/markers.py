"""Render and recover an identical visible marker on every candidate object."""
import cv2
import numpy as np


COLOR = (247, 11, 233)


def marked_image(rgb, points):
    if np.any(np.all(rgb == COLOR, axis=-1)):
        raise ValueError("Marker color already exists in the original RGB")
    rgb = rgb.copy()
    for point in points:
        cv2.drawMarker(rgb, tuple(point), (0, 0, 0), cv2.MARKER_CROSS, 13, 3, cv2.LINE_8)
        cv2.drawMarker(rgb, tuple(point), COLOR, cv2.MARKER_CROSS, 11, 1, cv2.LINE_8)
    if sorted(find_markers(rgb)) != sorted(map(tuple, points)):
        raise ValueError("Rendered markers overlap or cannot be recovered exactly")
    return rgb


def find_markers(rgb):
    mask = np.all(rgb == COLOR, axis=-1).astype(np.uint8)
    n, _, _, centers = cv2.connectedComponentsWithStats(mask)
    return [tuple(map(int, np.rint(center))) for center in centers[1:n]]


def locate_marker(markers, box):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    choices = [(x, y) for x, y in markers if x0 <= x < x1 and y0 <= y < y1]
    return min(choices, key=lambda p: (p[0]-(x0+x1)/2)**2+(p[1]-(y0+y1)/2)**2, default=None)
