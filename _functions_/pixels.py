import cv2

# Load the low-resolution image
low_res = cv2.imread('D:\\VS code\\Project\\static\\uploads\\sample1.jpg')

# Upscale the image using bicubic interpolation
scale_factor = 8
high_res = cv2.resize(low_res, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Save the output high-resolution image
cv2.imwrite('output1.jpg', high_res)

# Display the results
# cv2.imshow('Low-Resolution', low_res)
# cv2.imshow('High-Resolution', high_res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
