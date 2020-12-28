#Importing Required modules
import os
import cv2
import numpy as np
from PIL import Image
import pickle

#Setting path of base directory of the training module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#Setting path for folder of images in base directory
image_dir = os.path.join(BASE_DIR, "images")

#Haar Cascde Classifier - Used - Frontal Face
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
#Recognizer to train
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0 #unique id is alloted to each image
label_ids = {} #For each id a label is created
y_labels = [] #list to store all traing labels
x_train = [] #For labels+id

for root,dirs,files in os.walk(image_dir):
    for file in files:
        #Checking for images in the file
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label, path)
            #Alloting id and label
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert("L")#converting images to gray scale if not already done
            #Forming numpy array
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi_array = image_array[y:y+h, x:x+w]
                x_train.append(roi_array)
                y_labels.append(id_)

#Saving model
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)

#Traing model over formed dataset of images in form of numpy array
recognizer.train(x_train,np.array(y_labels))
#Saving trained results for future detection
recognizer.save("trainner.yml")

print("Training Complete!!!")