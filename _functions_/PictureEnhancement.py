import cv2
import numpy as np

image = cv2.imread('D:\\VS code\\Project\\static\\uploads\\sample1.jpg')  # Replace 'input_image.jpg' with the actual path to your input image

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized = cv2.equalizeHist(gray)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Apply image sharpening using a sharpening kernel
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, kernel)

cv2.imwrite('output_image.jpg', enhanced)  # Replace 'enhanced' with the output of your chosen enhancement technique
