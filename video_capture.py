import os
import cv2
from pre_processing import preprocess, getContouredImage

video_path = r"Input/vid_3.mp4"
out_contours_path = r"Output/contours_video_3.mp4"
os.makedirs(os.path.dirname(out_contours_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

# Try to read fps from input, fallback to 25 if unavailable
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or fps != fps:  # check for NaN
    fps = 25.0

clahe = cv2.createCLAHE(24, (8, 8))

def resize_frame(frame, resize_factor=0.7):
    return cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

def clahe_on_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe_local.apply(l)
    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return bgr2

# Read one frame to determine output size (after resize)
ret, sample = cap.read()
if not ret:
    raise RuntimeError("Input video contains no frames")

sample = resize_frame(sample, 0.5)
h, w = sample.shape[:2]
frame_size = (w, h)

# Setup VideoWriter for contour frames (mp4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_contours = cv2.VideoWriter(out_contours_path, fourcc, fps, frame_size, True)

# If you want to also write original video uncomment below:
# out_original = cv2.VideoWriter(r"Output/original_video.mp4", fourcc, fps, frame_size, True)

# Reset capture to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_index = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_frame(frame, 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    preprocessed = preprocess(gray)          # your existing function
    contours_img = getContouredImage(frame)  # your existing function

    # ensure contours_img is 3-channel BGR so VideoWriter can write it
    if len(contours_img.shape) == 2 or contours_img.shape[2] == 1:
        contours_bgr = cv2.cvtColor(contours_img, cv2.COLOR_GRAY2BGR)
    else:
        contours_bgr = contours_img

    # If sizes mismatch (just in case), resize contours to match
    if (contours_bgr.shape[1], contours_bgr.shape[0]) != frame_size:
        contours_bgr = cv2.resize(contours_bgr, frame_size, interpolation=cv2.INTER_AREA)

    # Show windows (optional)
    cv2.imshow("Underwater gate", frame)
    cv2.imshow("Underwater gray", gray)
    cv2.imshow("Preprocessed", preprocessed)
    cv2.imshow("Contours", contours_bgr)

    # Write contours frame to output video
    out_contours.write(contours_bgr)

    # Optionally write original frame
    # out_original.write(frame)

    frame_index += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out_contours.release()
# out_original.release()
cv2.destroyAllWindows()
print(f"Saved contours video to {out_contours_path}")