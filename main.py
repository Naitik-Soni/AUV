import cv2
from pipeline import gate_detection_pipeline
from video_writer import VideoWriter
from color_detection import color_detection_pipeline

def read_video(video_path, extras = None):
    cap = cv2.VideoCapture(video_path, extras) if extras is not None else cv2.VideoCapture(video_path)
    return cap

def resize_frame(frame, rf = 0.5*2):
    return cv2.resize(frame, None, None, fx=rf, fy=rf, interpolation=cv2.INTER_AREA)

def clahe_on_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe_local.apply(l)
    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return bgr2

# cap = read_video(r"Input/vid_1.mp4")
# cap = read_video(r"Input/vid_2.mp4")
# cap = read_video(r"Input/vid_3.mp4")
# cap = read_video(r"Input/vid_4.mp4")
# cap = read_video(r"Input/vid_5.mp4")
# cap = read_video(r"Input/vid_6.mp4")

cap = read_video(0, cv2.CAP_V4L2)

ret, frame = cap.read()

if not ret:
    raise FileNotFoundError("Video at destination not founc")

frame = resize_frame(frame)

# writer = VideoWriter(
#     output_path="output_gate_detection.mp4",
#     fps=25,
#     frame_size=(frame.shape[1], frame.shape[0])
# )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_frame(frame)
    frame = clahe_on_lab(frame)
    result, detection, position = gate_detection_pipeline(frame)
    color_detected = color_detection_pipeline(frame)

    print("Gate Detection Confidence:", detection, "Position:", position, "Color Detection:", color_detected)

    if result is None:
        continue

    cv2.imshow("Gate Detection", result)
    # writer.write(result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# writer.release()
cap.release()
cv2.destroyAllWindows()
