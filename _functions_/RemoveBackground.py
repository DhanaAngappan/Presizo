import numpy as np
import cv2

image = cv2.imread('D:\\VS code\\Project\\static\\uploads\\sample1.jpg') # Replace 'input_image.jpg' with the actual path to your input image
if image is None:
    print("Failed to load image")

# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mask = np.zeros(image.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

result = image * mask_2[:, :, np.newaxis]

cv2.imwrite('output_RemoveBackground.png', result)
