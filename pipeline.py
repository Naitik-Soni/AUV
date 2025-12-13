import cv2
import numpy as np

from green_channel_masking import green_channel_threshold
from noise_removal import clean_mask
from rectangle_detection import detect_rectangle_contours
from hough_lines import detect_hough_lines
from intersection import gate_decision
from draw_lines import draw_hough_lines

CONF_NONE   = 0
CONF_LOW    = 1
CONF_MEDIUM = 2
CONF_HIGH   = 3


def gate_detection_pipeline(frame):
    """
    Pipeline function:
    - takes a single frame
    - runs all CV stages
    - draws intermediate results
    - returns output frame + confidence
    """

    output = frame.copy()

    # --------------------------------------------------
    # 1. Green channel thresholding
    # --------------------------------------------------
    mask = green_channel_threshold(frame)

    # --------------------------------------------------
    # 2. Noise removal
    # --------------------------------------------------
    mask = clean_mask(mask)

    # --------------------------------------------------
    # 3. Rectangle detection (contours)
    # --------------------------------------------------
    contours, rectangles = detect_rectangle_contours(
        mask,
        frame.shape[0],
        frame.shape[1]
    )

    # Draw rectangles (GREEN)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    # --------------------------------------------------
    # 4. Hough line detection
    # --------------------------------------------------
    lines = detect_hough_lines(mask)

    # Draw Hough lines (RED)
    if lines is not None:
        # normalize line format: [(rho, theta), ...]
        flat_lines = []
        for l in lines:
            if isinstance(l[0], (list, tuple, np.ndarray)):
                flat_lines.append(tuple(l[0]))
            else:
                flat_lines.append(tuple(l))

        # draw_hough_lines(output, flat_lines)
    else:
        flat_lines = []

    # --------------------------------------------------
    # 5. Gate decision (SINGLE CALL)
    # --------------------------------------------------
    confidence = gate_decision(rectangles, flat_lines)

    # --------------------------------------------------
    # 6. Visualization label
    # --------------------------------------------------
    if confidence == CONF_HIGH:
        text = "GATE DETECTED"
        color = (0, 0, 255)
    elif confidence == CONF_MEDIUM:
        text = "GATE DETECTED"
        color = (0, 165, 255)
    elif confidence == CONF_LOW:
        text = "POSSIBLE GATE"
        color = (0, 255, 255)
    else:
        text = "NO GATE"
        color = (255, 255, 255)

    cv2.putText(
        output,
        text,
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    return output, confidence
