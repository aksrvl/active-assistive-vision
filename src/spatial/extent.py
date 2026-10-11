EXTENT_MARGIN = 40

def estimate_extent_from_roi(roi, frame_shape):
    x, y, w, h = roi
    frame_h, frame_w = frame_shape[:2]

    distances = {
        "left": x,
        "right": frame_w - (x + w),
        "top": y,
        "bottom": frame_h - (y + h)
    }

    extent = {
        "left": distances["left"] > EXTENT_MARGIN,
        "right": distances["right"] > EXTENT_MARGIN,
        "top": distances["top"] > EXTENT_MARGIN,
        "bottom": distances["bottom"] > EXTENT_MARGIN
    }

    return extent, distances

def update_extent(global_extent, observation_extent):
    updated_extent = {}

    for side in global_extent:
        updated_extent[side] = (global_extent[side] or observation_extent[side])

    return updated_extent

def is_extent_complete(extent):
    return all(extent.values())

def get_unknown_extent(extent):
    missing = []
    for side in extent:
        if not extent[side]:
            missing.append(side)

    return missing
