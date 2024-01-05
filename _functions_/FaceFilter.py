import cv2
import numpy as np

image = cv2.imread('D:\\VS code\\Project\\static\\uploads\\sample1.jpg')  # Replace 'input_image.jpg' with the actual path to your input image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


glasses = cv2.imread('D:\\VS code\\Project\\static\\img\\Glass2.png', -1) 

for (x, y, w, h) in faces:
  
    resized_glasses = cv2.resize(glasses, (w, int(h/2)))

   
    x_offset = x
    y_offset = y + int(h/4)

   
    for i in range(resized_glasses.shape[0]):
        for j in range(resized_glasses.shape[1]):
            if resized_glasses[i, j][2] != 0: 
                image[y_offset + i, x_offset + j] = resized_glasses[i, j][:3]

cv2.imwrite('output_image.jpg', image)

