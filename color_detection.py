import cv2
import numpy as np

MIN_SATURATION = 100
MIN_VALUE = 100

MAX_SATURATION = 255
MAX_VALUE = 255

def get_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def detect_red_color(image):
    hsv = get_hsv(image)

    lower1 = np.array([0, MIN_SATURATION, MIN_VALUE])
    upper1 = np.array([13, MAX_SATURATION, MAX_VALUE])

    lower2 = np.array([165, MIN_SATURATION, MIN_VALUE])
    upper2 = np.array([180, MIN_SATURATION, MIN_VALUE])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2

    return 1 if np.any(mask) else 0

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_orange_color(image):
    hsv = get_hsv(image)

    lower = np.array([10, MIN_SATURATION, MIN_VALUE])
    upper = np.array([22, MAX_SATURATION, MAX_VALUE])

    mask = cv2.inRange(hsv, lower, upper)

    return 1 if np.any(mask) else 0

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_yellow_color(image):
    hsv = get_hsv(image)

    lower = np.array([20, MIN_SATURATION, MIN_VALUE])
    upper = np.array([36, MAX_SATURATION, MAX_VALUE])

    mask = cv2.inRange(hsv, lower, upper)

    return 1 if np.any(mask) else 0

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_green_color(image):
    hsv = get_hsv(image)

    lower = np.array([33, MIN_SATURATION, MIN_VALUE])
    upper = np.array([87, MAX_SATURATION, MAX_VALUE])

    mask = cv2.inRange(hsv, lower, upper)

    return 1 if np.any(mask) else 0

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_blue_color(image):
    hsv = get_hsv(image)

    lower = np.array([90, MIN_SATURATION, MIN_VALUE])
    upper = np.array([140, MAX_SATURATION, MAX_VALUE])

    mask = cv2.inRange(hsv, lower, upper)

    return 1 if np.any(mask) else 0

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_detection_pipeline(image):
    return [
        detect_blue_color(image),
        detect_green_color(image),
        detect_yellow_color(image),
        detect_orange_color(image),
        detect_red_color(image)
    ]


# img = cv2.imread(r"balcony garden.jpg")
# f = 0.5
# img = cv2.resize(img, None, None, fx=f,fy=f)
# cv2.imshow("OG", img)

# detect_blue_color(img)
# detect_green_color(img)
# detect_yellow_color(img)
# detect_orange_color(img)
# detect_red_color(img)