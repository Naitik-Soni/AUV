import cv2

def green_channel_threshold(roi):
    b, g, r = cv2.split(roi)

    # Threshold on green channel
    _, thresh = cv2.threshold(g, 75, 255, cv2.THRESH_BINARY_INV)

    return thresh
