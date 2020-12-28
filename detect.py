#importing required modules
import numpy as np
import cv2
import pickle

#Haar Cascde Classifier - Used - Frontal Face
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle",'rb') as f:
    l = pickle.load(f)
    labels = {v:k for k,v in l.items()}

#Turning on webcam to collect images
cap = cv2.VideoCapture(0)

while(True):
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

        id_, conf = recognizer.predict(roi_gray )
        if conf>=50 :
            print(id_)
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,name,(x,y+h+30),font,1,(0,255,0),2,cv2.LINE_AA)
            print(name)
        else :
            cv2.putText(frame, "No Image Detected", (x, y + h + 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print("No Image Detected")

        #Representing frame by a blue box(bgr = 255,0,0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0.0), 5)

    print(conf)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



# CLosing webcam
cap.release()
cv2.destroyAllWindows()