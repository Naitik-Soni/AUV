import cv2

video_path = r"Input/Vid-2.mp4"
cap = cv2.VideoCapture(video_path)

clahe = cv2.createCLAHE(24, (8,8))

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_frame = clahe.apply(gray)
    lab_image = clahe_on_lab(frame)

    cv2.imshow("Underwater gate", frame)
    cv2.imshow("Underwater gray", gray)
    cv2.imshow("Underwater clahe", clahe_frame)
    cv2.imshow("Clahe on LAB", lab_image)

    cv2.imwrite(fr"Dataset_v1/Original/Gate-1 Frame {frame_index}.png", frame[:490, :])
    cv2.imwrite(fr"Dataset_v1/Gray/Gate-1 Frame {frame_index}.png", gray[:490, :])
    cv2.imwrite(fr"Dataset_v1/Contrasted/Gate-1 Frame {frame_index}.png", clahe_frame[:490, :])
    cv2.imwrite(fr"Dataset_v1/LAB/Gate-1 Frame {frame_index}.png", lab_image[:490, :])
    
    frame_index += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()