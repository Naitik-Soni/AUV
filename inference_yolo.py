import cv2
from ultralytics import YOLO

video_path = r"Input/Vid-2.mp4"
cap = cv2.VideoCapture(video_path)

model_path = r"P:\AUV\AUV\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)


def infer_yolo(frame_path):
    results = model.predict(source=frame_path, save=False, imgsz=640)

    img = cv2.imread(frame_path)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label text
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Overwrite same image with detections
    cv2.imwrite(frame_path, img)

    return "done"

def resize_frame(frame, resize_factor=0.7):
    return cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

def clahe_on_lab(bgr):
    # bgr: HxWx3 uint8
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return bgr2

frame_index = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = resize_frame(frame)
    lab_image = clahe_on_lab(frame)

    output_path = fr"Yolo_outputs/{frame_index} _2.jpg"
    cv2.imwrite(output_path, lab_image)

    infer_result = infer_yolo(output_path)

    frame_index += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()