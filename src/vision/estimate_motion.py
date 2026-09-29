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
        #cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

print("Total frames: ", frame_count)
print("Selected frames", len(selected_frames))

cap.release()

test_frame1 = cv.cvtColor(selected_frames[0], cv.COLOR_BGR2GRAY)
test_frame2 = cv.cvtColor(selected_frames[1], cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(test_frame1, None)
kp2, des2 = orb.detectAndCompute(test_frame2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k = 2)

threshold = 0.75
selected_matches = []

for best, second_best in matches:
    if best.distance < threshold * second_best.distance:
        selected_matches.append(best)

outcome_frame = cv.drawMatches(test_frame1, kp1, test_frame2, kp2, selected_matches[:30],None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('matches', outcome_frame)

cv.waitKey(0)

print("Raw matches:", len(matches))
print("Good matches:", len(selected_matches))

cv.destroyAllWindows()
