#importing required header files
import numpy as np
import cv2
import pickle

#Haar Cascde Classifier - Used - Frontal Face
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')

#Turning on webcam to collect images
cap = cv2.VideoCapture(0)
i=1

while(i<=250):
    # Capturing frame - shown by blue
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:#x,y,w,h = coordinates for frame
        print(x,y,h,w)
        #region of interest for gray image
        roi_gray = gray[y:y+h, x:x+w]
        #region of interest for color image
        roi_color = frame[y:y + h, x:x + w]
        #saving the cropped frame
        req_img = "./images/kunal/" + str(i) + ".png"
        cv2.imwrite(req_img, roi_gray)
        i+=1
        #Representing frame by a blue box(bgr = 255,0,0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0.0), 5)

        if i==250:
            break


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Releasing Webcam
cap.release()
cv2.destroyAllWindows()