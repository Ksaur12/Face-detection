import cv2
from random import randrange

#loading the pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#an image to detect face
#img = cv2.imread('img.jpg')
#we can give any video name instead of 0 i.e. "Movie.mkv"
webcam = cv2.VideoCapture(0)

while True:
	frame_read_successfully, frame = webcam.read()
	
	#converting into grayscale
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#this will output the coordinates of the rectangle
	face_coordinates = trained_face_data.detectMultiScale(gray_img)
	
	rect_color = (randrange(256), randrange(256), randrange(256))
	rect_width = 2
	
	#drawing the rectangle while looping through all the faces
	for (x_coor, y_coor, w, h) in face_coordinates:
		cv2.rectangle(frame, (x_coor, y_coor), (x_coor+w, y_coor+h ) , rect_color, rect_width)
	
	#we needed waitKey() to keep the window open otherwise it closes immediately
	cv2.imshow('Face Detector' , frame)
	key = cv2.waitKey(1) #milliseconds waiting
	
	if key==81 or key==113: #ascii of q and Q
		break

#release the video capture
webcam.release()

print('Done...')