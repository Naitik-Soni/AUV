import cv2
from green_channel_masking import *
from hough_lines import *
from intersection import *
from noise_removal import *
from rectangle_detection import *
from draw_lines import *

def gate_detection_pipeline(frame):
    output = frame.copy()

    # Threshold
    mask = green_channel_threshold(frame)

    # Clean
    mask = clean_mask(mask)

    # Contours
    contours, rectangles = detect_rectangle_contours(
        mask, frame.shape[0], frame.shape[1]
    )

    # Hough Lines
    lines = detect_hough_lines(mask)

    # Decision (keep as-is for now)
    gate_detected = gate_decision(rectangles, lines)

    # Draw rectangles (GREEN)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    # Draw Hough lines (RED)
    if lines is not None:
        flat_lines = [l[0] if isinstance(l[0], (list, tuple, np.ndarray)) else l for l in lines]
        draw_hough_lines(output, flat_lines)

    # Gate label
    if gate_detected:
        cv2.putText(
            output,
            "GATE DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    return output, gate_detected