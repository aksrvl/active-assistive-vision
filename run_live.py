import cv2 as cv
import time 
import os

CAMERA_INDEX = 1
MOTION_THRESHOLD = 50
MOTION_PIXEL_THRESHOLD = 0.06
STABLE_DURATION = 2.0

frame_count = 0
previous_time = time.perf_counter()
roi = None
previous_roi_gray = None
scene_state = "SELECT ROI"
stable_since = None

evidence_dir = "evidence"
os.makedirs(evidence_dir, exist_ok=True)
evidence_count = 0
snapshot_captured = False

cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

try:
    previous_time = time.perf_counter()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        evidence_frame = frame.copy()

        frame_count += 1
        height, width = frame.shape[:2]

        current_time = time.perf_counter()
        delta_time = current_time - previous_time
        fps = 1.0 / delta_time if delta_time > 0 else 0.
        previous_time = current_time

        motion_status = "SELECT ROI"

        if roi is None:
            scene_state = "SELECT ROI"
            stable_since = None
            previous_roi_gray = None

        else:
            x, y, w, h = roi

            roi_frame = frame[y:y+h, x:x+w]
            roi_gray = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
            roi_gray = cv.GaussianBlur(roi_gray, (5, 5), 0)

            if previous_roi_gray is None:
                motion_status = "INITIALIZING"
                scene_state = "OBSERVING"
                stable_since = None

            else:
                diff = cv.absdiff(previous_roi_gray, roi_gray)

                _, motion_mask = cv.threshold(
                    diff,
                    MOTION_THRESHOLD,
                    255,
                    cv.THRESH_BINARY
                )

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

                motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)
                motion_mask = cv.dilate(motion_mask, kernel, iterations=2)

                changed_pixels = cv.countNonZero(motion_mask)
                total_pixels = motion_mask.shape[0] * motion_mask.shape[1]

                motion_ratio = changed_pixels / total_pixels

                contours, _ = cv.findContours(motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                for contour in contours: 
                    if cv.contourArea(contour) < 100:
                        continue

                    mx, my, mw, mh = cv.boundingRect(contour)

                    cv.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (0,0,255), 2)

                if motion_ratio > MOTION_PIXEL_THRESHOLD:
                    motion_status = "MOTION"
                else:
                    motion_status = "NO MOTION"

                if motion_status == "MOTION":
                    stable_since = None
                    scene_state = "OBSERVING"
                    snapshot_captured = False
                else:
                    if stable_since is None:
                        stable_since = time.perf_counter()

                    elapsed = time.perf_counter() - stable_since

                    if elapsed >= STABLE_DURATION:
                        scene_state = "READY TO ANALYZE"
                    else: 
                        scene_state = "OBSERVING"
            
            previous_roi_gray = roi_gray.copy()

            if (scene_state == "READY TO ANALYZE" and not snapshot_captured):
                next_count = evidence_count + 1

                evidence_path = os.path.join(evidence_dir, f"scene_{next_count:03d}.jpg")
                saved = cv.imwrite(evidence_path, evidence_frame)

                if saved:
                    evidence_count = next_count
                    snapshot_captured = True
                    print(f"Evidence saved: {evidence_path}")
                else:
                    print("Failed to save evidence")
                
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv.putText(frame, f"{width}x{height}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"FPS: {fps}", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.putText(frame, f"Frame: {frame_count}", (0, 130), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.putText(frame, f"Status: {motion_status}", (0, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if motion_status == "NO MOTION" else (0, 0, 255), 2)
        cv.putText(frame, f"Scene: {scene_state}", (0,190), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv.imshow("Live Camera", frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("s"):
            selected_roi = cv.selectROI(
                "Live Camera",
                frame,
                showCrosshair=True,
                fromCenter=False
            )
            x, y, w, h = selected_roi
            previous_roi_gray = None

            if w > 0 and h > 0:
                roi = (x, y, w, h)
                previous_roi_gray = None
                stable_since = None
                snapshot_captured = False
                scene_state = "OBSERVING"
                
                print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
            else:
                print("ROI selection cancelled")

        elif key == ord("r"):
            roi = None
            previous_roi_gray = None
            stable_since = None
            snapshot_captured = False
            scene_state = "SELECT ROI"
            motion_status = "SELECT ROI"
            print("ROI reset")

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv.destroyAllWindows()
