def choose_exploration_target(unknown_extent, distances):
    return min(unknown_extent, key=lambda side: distances[side])

def plan_camera_action(target):
    camera_action = {}

    if target == "left":
        camera_action['direction'] = 'left'
        camera_action['magnitude'] = 'small'
        camera_action["goal"] = 'reveal_left_boundary'
        camera_action['hint'] = 'move left'

    elif target == "right":
        camera_action['direction'] = 'right'
        camera_action['magnitude'] = 'small'
        camera_action["goal"] = 'reveal_right_boundary'
        camera_action['hint'] = 'move right'

    elif target == "top":
        camera_action['direction'] = 'top'
        camera_action['magnitude'] = 'small'
        camera_action["goal"] = 'reveal_top_boundary'
        camera_action['hint'] = 'raise camera'

    elif target == "bottom":
        camera_action['direction'] = 'bottom'
        camera_action['magnitude'] = 'small'
        camera_action["goal"] = 'reveal_bottom_boundary'
        camera_action['hint'] = 'lower camera'

    return camera_action

def evaluate_camera_action(camera_action, motion):
    direction_mapping = {
        "left": "left",
        "right": "right",
        "top": "up",
        "bottom": "down"
    }
    opposites = {
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
        "forward": "backward",
        "backward": "forward"
    }
    expected = direction_mapping[camera_action["direction"]]
    observed = motion["translation_direction"]
    if not motion["reliable"]:
        status = "UNCERTAIN"
        reason = "Camera motion is unreliable"

    elif expected == observed:
        status = "FOLLOWED"
        reason = "Camera moved in the requested direction"

    elif observed == opposites[expected]:
        status = "WRONG_DIRECTION"
        reason = "Camera moved in the wrong direction"

    else:
        status = "OFF_AXIS"
        reason = "Camera moved along a different axis"
    
    return {
        "status": status,
        "expected": expected,
        "observed": observed,
        "reason": reason
    }

