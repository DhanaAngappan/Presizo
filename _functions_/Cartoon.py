import cv2

image = cv2.imread('D:\\VS code\\Project\\static\\uploads\\sample1.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 9, 75, 75)

edges = cv2.Canny(filtered, 30, 150)


colors = cv2.medianBlur(image, 15)


cartoon = cv2.bitwise_and(colors, colors, mask=edges)

cv2.imwrite('output_image.jpg', cartoon)
