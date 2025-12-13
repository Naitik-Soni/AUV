import numpy as np

CONF_NONE   = 0
CONF_LOW    = 1
CONF_MEDIUM = 2
CONF_HIGH   = 3


def gate_decision(rectangles, lines, px_thresh=5):
    """
    Logic:
    - Rectangle only                        -> LOW
    - Rectangle + 2 vertical lines near it  -> MEDIUM
    - Rectangle + 2 vertical + 1 horizontal -> HIGH
    - Otherwise                             -> NONE
    """

    # --------------------------------------------------
    # 0. Basic validation
    # --------------------------------------------------
    if rectangles is None or len(rectangles) == 0:
        return CONF_NONE

    if lines is None:
        lines = []

    # --------------------------------------------------
    # 1. Pick the most likely gate rectangle
    #    (tallest one)
    # --------------------------------------------------
    x, y, w, h = max(rectangles, key=lambda r: r[3])

    left_x  = x
    right_x = x + w
    top_y   = y

    # --------------------------------------------------
    # 2. Classify lines
    # --------------------------------------------------
    vertical_lines = []
    horizontal_lines = []

    for rho, theta in lines:
        angle = theta * 180 / np.pi

        if angle < 5 or angle > 175:
            vertical_lines.append((rho, theta))
        elif 75 < angle < 105:
            horizontal_lines.append((rho, theta))

    # --------------------------------------------------
    # 3. Rectangle exists → LOW confidence
    # --------------------------------------------------
    confidence = CONF_LOW

    # --------------------------------------------------
    # 4. Check vertical line proximity
    # --------------------------------------------------
    vertical_hits = 0

    for rho, theta in vertical_lines:
        # distance from rectangle vertical sides
        if abs(left_x * np.cos(theta) - rho) < px_thresh:
            vertical_hits += 1
        elif abs(right_x * np.cos(theta) - rho) < px_thresh:
            vertical_hits += 1

    # --------------------------------------------------
    # 5. Rectangle + 2 vertical lines → MEDIUM
    # --------------------------------------------------
    if vertical_hits < 2:
        return confidence

    confidence = CONF_MEDIUM

    # --------------------------------------------------
    # 6. Check horizontal line near top edge
    # --------------------------------------------------
    for rho, theta in horizontal_lines:
        if abs(top_y * np.sin(theta) - rho) < px_thresh:
            return CONF_HIGH

    # --------------------------------------------------
    # 7. No horizontal → MEDIUM
    # --------------------------------------------------
    return confidence
