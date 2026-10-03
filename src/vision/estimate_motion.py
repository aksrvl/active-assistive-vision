import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = "./data/raw/sofa_video.MOV"
CALIBRATION_PATH = "./data/calibration/camera_calibration.npz"

FRAME_STEP = 10
LOWE_THRESHOLD = 0.75
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

    inlier_mask = pose_mask.ravel() != 0

    inlier_points1 = undistorted_points1[inlier_mask]
    inlier_points2 = undistorted_points2[inlier_mask]

    pose_inlier_ratio = pose_inliers / essential_inliers if essential_inliers != 0 else 0

    return R, t, len(selected_matches), essential_inliers, pose_inliers, pose_inlier_ratio, inlier_points1, inlier_points2


keyframe_index = 0
candidate_index = 1

reference_frame = selected_frames[0].copy()
roi = cv.selectROI("Select TRR", reference_frame)
cv.destroyWindow("Select TRR")
print(f"Selected TRR: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

cv.rectangle(reference_frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,255,0), 3)
cv.imshow("TRR", reference_frame)
cv.waitKey(0)
cv.destroyAllWindows()

R_global = np.eye(3)
t_global = np.zeros((3, 1))

camera_positions = [np.zeros(3)]

while candidate_index < len(selected_frames):
    frame1 = selected_frames[keyframe_index]
    frame2 = selected_frames[candidate_index]

    R, t, matches_count, essential_inliers, pose_inliers, pose_inlier_ratio, inlier_points1, inlier_points2 = estimate_relative_pose(
        frame1,
        frame2,
        camera_matrix,
        dist_coeffs
    )

    if pose_inliers >= MIN_POSE_INLIERS and pose_inlier_ratio >= MIN_POSE_INLIER_RATIO:
        print(
            f"{keyframe_index} -> {candidate_index} | "
            f"pose: {pose_inliers}/{essential_inliers} | "
            f"ratio: {pose_inlier_ratio:.2f} | ACCEPT"
        )

        if(keyframe_index == 0):
            trr_mask = ((inlier_points1[:, 0] >= roi[0]) & 
                        (inlier_points1[:,0]<=roi[0]+roi[2]) & 
                        (inlier_points1[:, 1]>=roi[1]) & 
                        (inlier_points1[:,1]<=roi[1]+roi[3]))
          
            print(trr_mask.shape)
            
            trr_points_frame1 = inlier_points1[trr_mask]
            trr_points_frame2 = inlier_points2[trr_mask]

            frame1_vis = frame1.copy()
            frame2_vis = frame2.copy()

            for point1, point2 in zip(trr_points_frame1, trr_points_frame2):
                cv.circle(frame1_vis, (int(point1[0]), int(point1[1])), 5, (0, 255, 0), -1)
                cv.circle(frame2_vis, (int(point2[0]), int(point2[1])), 5, (0, 255, 0), -1)
            cv.imshow("Frame 1 vis", frame1_vis)
            cv.imshow("Frame 2 vis", frame2_vis)
            cv.waitKey(0)
            cv.destroyAllWindows()

            print("TRR tracked points:", len(trr_points_frame1))

        t_global = R @ t_global + t
        R_global = R @ R_global

        camera_center = -R_global.T @ t_global
        camera_positions.append(camera_center.flatten())

        keyframe_index = candidate_index

    else:
        print(
            f"{keyframe_index} -> {candidate_index} | "
            f"pose: {pose_inliers}/{essential_inliers} | "
            f"ratio: {pose_inlier_ratio:.2f} | SKIP"
        )

    candidate_index += 1

camera_positions = np.array(camera_positions)
plt.plot(
    camera_positions[:, 0],
    camera_positions[:, 2],
    marker="o"
)

plt.xlabel("X (arbitrary scale)")
plt.ylabel("Z (arbitrary scale)")
plt.title("Estimated Camera Trajectory (Top View)")
plt.axis("equal")
plt.grid()

plt.show()

