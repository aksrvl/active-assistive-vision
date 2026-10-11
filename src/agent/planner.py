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


