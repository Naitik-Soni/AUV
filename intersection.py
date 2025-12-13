import numpy as np

def classify_hough_lines(lines):
    vertical = []
    horizontal = []

    for rho, theta in lines:
        angle = theta * 180 / np.pi

        if angle < 5 or angle > 175:
            vertical.append((rho, theta))
        elif 75 < angle < 105:
            horizontal.append((rho, theta))

    return vertical, horizontal

def find_vertical_gate_posts(rectangles, min_height_ratio=0.6):
    if len(rectangles) < 2:
        return None

    rectangles = sorted(rectangles, key=lambda r: r[0])  # sort by x

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            x1, y1, w1, h1 = rectangles[i]
            x2, y2, w2, h2 = rectangles[j]

            height_ratio = min(h1, h2) / max(h1, h2)
            x_gap = abs(x2 - x1)

            if height_ratio > min_height_ratio and x_gap > w1:
                return rectangles[i], rectangles[j]

    return None

def horizontal_line_between_posts(horizontal_lines, left_rect, right_rect):
    lx, ly, lw, lh = left_rect
    rx, ry, rw, rh = right_rect

    x_left = lx + lw
    x_right = rx

    for rho, theta in horizontal_lines:
        # y = rho / sin(theta)
        y = int(rho / np.sin(theta))

        if ly < y < ly + lh:
            return True

    return False

def gate_decision(rectangles, lines):
    if len(rectangles) < 2 or len(lines) < 2:
        return 0

    vertical_lines, horizontal_lines = classify_hough_lines(lines)

    if len(horizontal_lines) == 0:
        return 0

    posts = find_vertical_gate_posts(rectangles)

    if posts is None:
        return 0

    left_post, right_post = posts

    if not horizontal_line_between_posts(horizontal_lines, left_post, right_post):
        return 0
    return 1

