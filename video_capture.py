import cv2
from pre_processing import preprocess, getContouredImage

video_path = r"Input/vid_1.mp4"
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

    frame = resize_frame(frame, 0.6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 7)
    # blurred = cv2.bilateralFilter(gray, 3, 50, 50)
    # clahe_frame = clahe.apply(gray)
    # lab_image = clahe_on_lab(frame)
    # canny = cv2.Canny(blurred, 60, 120)
    # canny_clahe_frame = cv2.Canny(clahe_frame, 64, 128)
    preprocessed = preprocess(gray)
    contours = getContouredImage(frame)

    cv2.imshow("Underwater gate", frame)
    cv2.imshow("Underwater gray", gray)
    cv2.imshow("Preprocessed", preprocessed)
    cv2.imshow("Contours", contours)
    # cv2.imshow("Canny", canny)

    # cv2.imshow("Underwater clahe", clahe_frame)
    # cv2.imshow("Clahe on LAB", lab_image)
    # cv2.imshow("Canny on CLAHE", canny_clahe_frame)

    # cv2.imwrite(fr"Dataset_v1/Original/Gate-1 Frame {frame_index}.png", frame[:490, :])
    # cv2.imwrite(fr"Dataset_v1/Gray/Gate-1 Frame {frame_index}.png", gray[:490, :])
    # cv2.imwrite(fr"Dataset_v1/Contrasted/Gate-1 Frame {frame_index}.png", clahe_frame[:490, :])
    # cv2.imwrite(fr"Dataset_v1/LAB/Gate-1 Frame {frame_index}.png", lab_image[:490, :])
    
    frame_index += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()