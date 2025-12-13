import cv2
import numpy as np

def detect_hough_lines(mask):
    edges = cv2.Canny(mask, 60, 120)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    valid_lines = []

    if lines is None:
        return valid_lines

    for line in lines:
        rho, theta = line[0]
        angle = abs(theta * 180 / np.pi)

        # Nearly vertical or nearly horizontal
        if angle < 5 or abs(angle - 90) < 15:
            valid_lines.append((rho, theta))

    return valid_lines
