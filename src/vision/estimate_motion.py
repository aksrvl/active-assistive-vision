import cv2 as cv
import numpy as np

VIDEO_PATH = "./data/raw/sofa_video.MOV"
CALIBRATION_PATH = "./data/calibration/camera_calibration.npz"

FRAME_STEP = 10
LOWE_THRESHOLD = 0.75
MIN_POSE_INLIERS = 50
MIN_POSE_INLIER_RATIO = 0.7
EXTENT_MARGIN = 40

def estimate_relative_pose(frame1, frame2, camera_matrix, dist_coeffs):
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    selected_matches = []

    for best, second_best in matches:
        if best.distance < LOWE_THRESHOLD * second_best.distance:
            selected_matches.append(best)


    points1 = []
    points2 = []

    for match in selected_matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)

    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    undistorted_points1 = cv.undistortPoints(
        points1,
        camera_matrix,
        dist_coeffs,
        R=None,
        P=camera_matrix
    ).reshape(-1, 2)

    undistorted_points2 = cv.undistortPoints(
        points2,
        camera_matrix,
        dist_coeffs,
        R=None,
        P=camera_matrix
    ).reshape(-1, 2)


    E, essential_mask = cv.findEssentialMat(
        undistorted_points1,
        undistorted_points2,
        camera_matrix,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    essential_inliers = np.count_nonzero(essential_mask)

    pose_inliers, R, t, pose_mask = cv.recoverPose(
        E,
        undistorted_points1,
        undistorted_points2,
        camera_matrix,
        mask=essential_mask
    )

    pose_inlier_ratio = pose_inliers / essential_inliers if essential_inliers != 0 else 0

    return R, t, pose_inliers, pose_inlier_ratio

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

def choose_exploration_target(unknown_extent, distances):
    return min(unknown_extent, key=lambda side: distances[side])
        


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
        if(keyframe_index == 0):
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
                target = choose_exploration_target(unknown, observation_distances)
                print("Missing extent:", unknown)  
                print("Exploration target:", target)

        t_global = R @ t_global + t
        R_global = R @ R_global

        keyframe_index = candidate_index

    candidate_index += 1
