import cv2

def detect_rectangle_contours(mask, h_i, w_i):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    rect_contours = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(cnt)

        sides = len(approx)
        area = cv2.contourArea(cnt)

        if sides == 4 and area >= (h_i*w_i)*0.05:
            rect_contours.append((x, y, w, h))

    return contours, rect_contours
