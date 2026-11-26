"""Select the merged patch at the point and its region-mean control."""


def select_features(grid, x, y, width, height):
    rows, cols = grid.shape[:2]
    row, col = int((y + .5) * rows / height), int((x + .5) * cols / width)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("Point fell outside its merged visual grid")
    return {"point": grid[row, col].cpu(), "mean": grid.mean((0, 1)).cpu()}, (row, col)
