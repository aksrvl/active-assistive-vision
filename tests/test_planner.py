from src.agent.planner import evaluate_camera_action


def test_action_followed():
    camera_action = {"direction": "left"}
    motion = {"translation_direction": "left", "reliable": True}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "FOLLOWED"
    assert result["expected"] == "left"
    assert result["observed"] == "left"


def test_wrong_direction():
    camera_action = {"direction": "left"}
    motion = {"translation_direction": "right", "reliable": True}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "WRONG_DIRECTION"


def test_off_axis():
    camera_action = {"direction": "left"}
    motion = {"translation_direction": "forward", "reliable": True}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "OFF_AXIS"


def test_unreliable_motion():
    camera_action = {"direction": "left"}
    motion = {"translation_direction": "left", "reliable": False}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "UNCERTAIN"


def test_top_direction():
    camera_action = {"direction": "top"}
    motion = {"translation_direction": "up", "reliable": True}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "FOLLOWED"
    assert result["expected"] == "up"


def test_bottom_direction():
    camera_action = {"direction": "bottom"}
    motion = {"translation_direction": "down", "reliable": True}

    result = evaluate_camera_action(camera_action, motion)

    assert result["status"] == "FOLLOWED"
    assert result["expected"] == "down"