import cv2 as cv
import numpy as np
from src.vision.pose import estimate_relative_pose
from src.vision.motion import estimate_camera_motion

from src.spatial.extent import (
    estimate_extent_from_roi,
    update_extent,
    is_extent_complete,
    get_unknown_extent
)

from src.agent.planner import (
    choose_exploration_target,
    plan_camera_action
)

VIDEO_PATH = "./data/raw/sofa_video.MOV"
CALIBRATION_PATH = "./data/calibration/camera_calibration.npz"

FRAME_STEP = 10
MIN_POSE_INLIERS = 50
MIN_POSE_INLIER_RATIO = 0.7

selected_frames = []

cap = cv.VideoCapture(VIDEO_PATH)

calibration = np.load(CALIBRATION_PATH)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_STEP == 0:
        selected_frames.append(frame)

cap.release()

if not selected_frames:
    raise RuntimeError("No frames were loaded from the video")


keyframe_index = 0
candidate_index = 1

reference_frame = selected_frames[0].copy()

roi = cv.selectROI("Select TRR", reference_frame)
cv.destroyWindow("Select TRR")

trr_extent, extent_distances = estimate_extent_from_roi(roi, reference_frame.shape)

print("Selected TRR:", roi)
print("TRR extent:", trr_extent)

R_global = np.eye(3)
t_global = np.zeros((3, 1))

target = None

while candidate_index < len(selected_frames):
    frame1 = selected_frames[keyframe_index]
    frame2 = selected_frames[candidate_index]

    R, t, pose_inliers, pose_inlier_ratio = estimate_relative_pose(
        frame1,
        frame2,
        camera_matrix,
        dist_coeffs
    )

    if pose_inliers >= MIN_POSE_INLIERS and pose_inlier_ratio >= MIN_POSE_INLIER_RATIO:
        motion = estimate_camera_motion(R, t)
        print("Camera motion:", motion)

        if not is_extent_complete(trr_extent):
            new_roi = cv.selectROI("Select TRR in new view", frame2)
            cv.destroyWindow("Select TRR in new view")
            observation_extent, observation_distances = estimate_extent_from_roi(
                new_roi,
                frame2.shape
            )
            print("Previous extent:", trr_extent)
            print("Observation extent:", observation_extent)
            print("Extent distances:", extent_distances)
            print("Observation distances:", observation_distances)

            trr_extent = update_extent(
                trr_extent,
                observation_extent
            )

            print("Updated extent:", trr_extent)     

            if is_extent_complete(trr_extent):
                print("EXTENT_COMPLETE")   
            else:
                print("NEED_MORE")
                unknown = get_unknown_extent(trr_extent)
                if target is None or trr_extent[target]:
                    target = choose_exploration_target(unknown, observation_distances)

                camera_action = plan_camera_action(target)

                print("Missing extent:", unknown)
                print("Exploration target:", target)
                print("Camera action:", camera_action)

        t_global = R @ t_global + t
        R_global = R @ R_global

        keyframe_index = candidate_index

    candidate_index += 1
