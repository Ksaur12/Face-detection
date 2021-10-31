import cv2
import random

#loading the pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#an image to detect face
img = cv2.imread('img.jpg')

#converting into grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#this will output the coordinates of the rectangle
face_coordinates = trained_face_data.detectMultiScale(gray_img)

rect_color = (random.randrange(255), random.randrange(255), random.randrange(255))
rect_width = 2

#drawing the rectangle while looping through all the faces
for (x_coor, y_coor, w, h) in face_coordinates:
	cv2.rectangle(img, (x_coor, y_coor), (x_coor+w, y_coor+h ) , rect_color, rect_width)

#we needed waitKey() to keep the window open otherwise it closes immediately
cv2.imshow('Face Detector', img)
key = cv2.waitKey()

print('Done...')