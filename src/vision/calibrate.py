import glob
import cv2 as cv
import numpy as np

images = glob.glob("./data/calibration/*.jpg")

print(len(images))

object_points = np.zeros((9*6, 3), dtype=np.float32)
print(object_points.shape)

object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

all_object_points = []
all_image_points = []

print(object_points)

for filename in images:
    image = cv.imread(filename)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    found, corners = cv.findChessboardCornersSB(
        gray,
        (9, 6)
    )
    if found:
        all_object_points.append(object_points)
        all_image_points.append(corners)

    print(filename, found)
    print("Image size:",image_size)

cv.drawChessboardCorners(image, (9, 6), corners, found)


cv.imshow("Detected corners", image)
cv.waitKey(0)
cv.destroyAllWindows()

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    all_object_points,
    all_image_points,
    image_size,
    None,
    None
)

reprojection_errors = []

for i in range(len(all_object_points)):
    projected_points, _ = cv.projectPoints(
        all_object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    detected = all_image_points[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    distances = np.linalg.norm(detected - projected, axis=1)
    view_error = np.mean(distances)
    reprojection_errors.append(view_error)

    print(f"View {i}: {view_error:.4f} px")

print(all_image_points[0].shape)
print(projected_points.shape)

mean_error = np.mean(reprojection_errors)
print(f"Mean reprojection error: {mean_error:.4f} px")

np.savez("./data/calibration/camera_calibration.npz", camera_matrix = camera_matrix, dist_coeffs = dist_coeffs, image_size = image_size)
calibration = np.load("./data/calibration/camera_calibration.npz")

print(calibration["camera_matrix"])
print(calibration["dist_coeffs"])
print(calibration["image_size"])
