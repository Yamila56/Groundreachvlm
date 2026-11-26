"""Prompt legend for RGB and its aligned map."""




LEGEND = (
    "Image 1 is the RGB scene. Image 2 is a spatial map aligned pixel-for-pixel with it. "
    "Small black marks in image 1 identify the points to check on the object surfaces. "
    "A point is reachable if its map color is strongly saturated "
    "(largest minus smallest RGB component is at least 83 on the 0–255 scale). "
    "Check the point on the object named by the question. Respond with exactly Yes or No."
)
