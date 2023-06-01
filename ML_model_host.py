from keras.models import load_model
import numpy as np
from PIL import ImageGrab 
import cv2
import os
import os.path
import glob
import imutils
#update on 25 April 2023

print ("Face Recognition: Deploy model")
input = input("choose model a-f: ")
input = str(input)
model_list = {"a":"F:/Work Folder/ML/best/test1.h5",
              "b":"F:/Work Folder/ML/best/test4.h5",
              "c":"F:/Work Folder/ML/best/test6.h5",
              "d":"F:/Work Folder/ML/best/test7-1.h5",
              "e":"F:/Work Folder/ML/best/test7-2.h5",
              "f":"F:/Work Folder/ML/best/test7XXX.h5"}

selected = (model_list[input])
final = (os.path.relpath(selected, "F:/Work Folder/ML/best/"))
print ("selected model: "+ final)
print ("")
face_cascade = cv2.CascadeClassifier('D:/Programs/python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
recognizer = load_model(selected)
recognizer.summary()


while True:
    img = ImageGrab.grab()
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(40,40))

    for (i,(x,y,w,h)) in enumerate (faces):

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
        
        roi_gray = np.array(roi_gray)
        roi_gray = cv2.resize(roi_gray,(40,40))
        roi_gray = np.expand_dims(roi_gray, axis = 0)
        roi_gray = roi_gray[:, :, :, np.newaxis]
        #predict the result
        result = recognizer.predict(roi_gray)   
        result_class = result.argmax(axis=1)
        txt= str(result_class)
        percent = np.nanmax(result)
        percents = str(percent*100)

        dict = {"[0]":"eunha","[1]":"sinb","[2]":"sowon","[3]":"umji","[4]":"yerin","[5]":"yuju"}
        s = np.array2string(result_class) 
      
        member = (dict[s])   
        print ("Prediction: "+member+" "+percents+"%")
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,65,152),2)
        cv2.putText(frame, member, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
    cv2.imshow("screen", frame)
    
    if cv2.waitKey(1)==27:
       break
cv2.destroyAllWindows