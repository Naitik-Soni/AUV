import numpy as np
import cv2

def draw_hough_lines(frame, lines, color=(0, 0, 255)):
    if lines is None:
        return frame

    h, w = frame.shape[:2]

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        # Extend line fully across image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

    return frame
