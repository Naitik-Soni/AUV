import cv2

img_path = r"P:\AUV\AUV\Dataset_v1\Original\Gate-1 Frame 14.png"
image = cv2.imread(img_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)

cv2.waitKey(0)
cv2.destroyAllWindows()