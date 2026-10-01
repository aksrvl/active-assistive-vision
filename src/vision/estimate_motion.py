import cv2 as cv
import numpy as np

VIDEO_PATH = "./data/raw/sofa_video.MOV"
CALIBRATION_PATH = "./data/calibration/camera_calibration.npz"

FRAME_STEP = 10
LOWE_THRESHOLD = 0.75

selected_frames = []

cap = cv.VideoCapture(VIDEO_PATH)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_STEP == 0:
        selected_frames.append(frame)

cap.release()

frame1 = cv.cvtColor(selected_frames[0], cv.COLOR_BGR2GRAY)
frame2 = cv.cvtColor(selected_frames[5], cv.COLOR_BGR2GRAY)

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


calibration = np.load(CALIBRATION_PATH)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]


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


rvec, _ = cv.Rodrigues(R)
rotation_angle = np.degrees(np.linalg.norm(rvec))


print(f"Matches after Lowe filtering: {len(selected_matches)}")
print(f"Essential matrix inliers: {essential_inliers}")
print(f"Pose inliers: {pose_inliers}")

print("\nRotation:")
print(R)

print("\nTranslation direction:")
print(t.ravel())

print(f"\nRotation angle: {rotation_angle:.2f} degrees")