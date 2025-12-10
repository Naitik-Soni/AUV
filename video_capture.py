import cv2

video_path = r"Input/vid_1.mp4"
cap = cv2.VideoCapture(video_path)

clahe = cv2.createCLAHE(24, (8,8))

def resize_frame(frame, resize_factor=0.7):
    return cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

frame_index = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = resize_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_frame = clahe.apply(gray)

    cv2.imshow("Underwater gate", frame)
    cv2.imshow("Underwater gray", gray)
    cv2.imshow("Underwater clahe", clahe_frame)

    cv2.imwrite(fr"Dataset/Original/Frame {frame_index}.png", frame[:490, :])
    cv2.imwrite(fr"Dataset/Gray/Frame {frame_index}.png", gray[:490, :])
    cv2.imwrite(fr"Dataset/Contrasted/Frame {frame_index}.png", clahe_frame[:490, :])
    frame_index += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()