import cv2 as cv

FRAME_STEP = 10

selected_frames = []

cap = cv.VideoCapture("./data/raw/sofa_video.MOV")

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)

print(f"Resolution: {width}x{height}, FPS: {fps}")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    frame_count += 1

    if frame_count%FRAME_STEP == 0:
        selected_frames.append(frame)
        cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

print("Total frames: ", frame_count)
print("Selected frames", len(selected_frames))

cap.release()
cv.destroyAllWindows()
