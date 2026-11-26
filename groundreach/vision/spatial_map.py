"""Shared RGB/map crops and the pixel-rule workspace mask."""

from PIL import ImageChops


def workspace_mask(spatial):
    r, g, b = spatial.split()
    brightest = ImageChops.lighter(ImageChops.lighter(r, g), b)
    darkest = ImageChops.darker(ImageChops.darker(r, g), b)
    return ImageChops.subtract(brightest, darkest).point(
        [255 if value >= 83 else 0 for value in range(256)]
    )


def shared_point_crop(points, width, height):
    xs, ys = zip(*points)
    return (max(0, min(xs)-48), max(0, min(ys)-48), min(width, max(xs)+49), min(height, max(ys)+49))
