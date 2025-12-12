import cv2

d = 3
sigma = 60

def getContouredImage(frame):
    image = frame.copy()
    h_i, w_i = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = blur_bilateral(gray)
    thresh = canny(blur)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    for cnt in contours:

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)
        # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 72, 72), 4)

        # Shape classification
        sides = len(approx)
        area = cv2.contourArea(cnt)

        if sides == 4 and area >= (h_i*w_i)*0.05:
            shape = "Gate"
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image

def canny(image):
    return cv2.Canny(image, 60, 120)

def blur_bilateral(image):
    return cv2.bilateralFilter(image, d, sigma, sigma)

def preprocess(image):
    image = blur_bilateral(image)
    image = canny(image)

    return image