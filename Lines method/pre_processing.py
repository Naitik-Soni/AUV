import cv2
import numpy as np
import math

d = 3
sigma = 60

def getHoughLinesP(image):
    # Returns (x1, y1, x2, y2)
    return cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi/180,
        threshold=70,
        minLineLength=50,
        maxLineGap=40
    )

def applyCanny(image):
    return cv2.Canny(image, 60, 120)

def blur_bilateral(image):
    return cv2.bilateralFilter(image, d, sigma, sigma)

def process(frame):
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = blur_bilateral(gray)
    canny = applyCanny(blur)

    lines = getHoughLinesP(canny)

    h, w = image.shape[:2]
    angle_thresh_deg = 3
    min_length = h * 0.5   # half height

    if lines is None or len(lines) == 0:
        return image

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # ---- Line angle ----
        dx = x2 - x1
        dy = y2 - y1

        angle_deg = abs(math.degrees(math.atan2(dy, dx)))

        # vertical lines = angle near 90° 
        if abs(angle_deg - 90) > angle_thresh_deg:
            continue

        # ---- Line length ----
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        # --- Draw ---
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    return image