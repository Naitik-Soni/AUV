import cv2
import numpy as np

image_path = r"../Input/arena.png"
image = cv2.imread(image_path)

# ------------------------------
# Optional CLAHE on LAB
# ------------------------------
def clahe_on_lab(bgr, enable=True):
    if not enable:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Step 1: Original
processed = image.copy()

# Step 2: CLAHE (disable if not required)
processed = clahe_on_lab(processed, enable=True)

# Step 3: Bilateral filter (edge-preserving)
processed = cv2.bilateralFilter(processed, d=7, sigmaColor=75, sigmaSpace=75)

# ------------------------------
# Green Detection (BGR-based)
# ------------------------------
b, g, r = cv2.split(processed)

# Green dominance logic
green_mask = np.zeros_like(g)
green_mask[(g > 80) & (g > r + 15) & (g > b + 15)] = 255

# ------------------------------
# Morphological cleanup
# ------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# ------------------------------
# Area Filtering & Detection
# ------------------------------
contours, _ = cv2.findContours(
    green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 3000:  # tune based on image size
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

# ------------------------------
# Display
# ------------------------------
cv2.imshow("Original", image)
cv2.imshow("Processed (CLAHE + Bilateral)", processed)
cv2.imshow("Green Mask", green_mask)
cv2.imshow("Final Detection", output)

cv2.waitKey(0)
cv2.destroyAllWindows()