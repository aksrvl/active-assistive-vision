import numpy as np
import cv2 as cv

def estimate_camera_motion(R, t):
    # 1. Camera center displacement
    camera_displacement = -R.T@t

    dx = camera_displacement[0][0]
    dy = camera_displacement[1][0]
    dz = camera_displacement[2][0]

    # 2. Normalized translation direction
    vector_length = np.sqrt(dx**2 + dy**2 + dz**2)
    normal_direction = camera_displacement/vector_length

    dx_norm = normal_direction[0][0]
    dy_norm = normal_direction[1][0]
    dz_norm = normal_direction[2][0]

    # 3. Dominant translation axis
    max_direction = np.argmax(np.abs([dx_norm, dy_norm, dz_norm]))
    directions = {
        0: ("right", "left"),
        1: ("down", "up"),
        2: ("forward", "backward")
    }
    direction=None
    match max_direction:
        case 0:
            if dx_norm>0:
                direction = directions[0][0]
            else:
                direction = directions[0][1]
        case 1:
            if dy_norm>0:
                direction = directions[1][0]
            else:
                direction = directions[1][1]
        case 2:
            if dz_norm>0:
                direction = directions[2][0]
            else:
                direction = directions[2][1]

    # 4. Rotation angle in degrees
    rotation_vector, _ = cv.Rodrigues(R)
    rotation_deg = np.linalg.norm(rotation_vector) * 180 / np.pi

    # 5. Return structured motion information
    return  {
        "translation_direction": direction,
        # "dominant_axis": dominant_axis,
        "normalized_translation": normal_direction.flatten(),
        "rotation_degrees": rotation_deg
    }
