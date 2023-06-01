from keras.models import load_model
import numpy as np
from PIL import ImageGrab 
from PIL import Image 
import cv2
import os
import os.path
import glob
from mss import mss
import cv2
import imutils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import time

print ("Face Recognition: Launcher")
inpt = input("choose mode: 2==webcam(),1==auto(),0==onscreen()")
inpt = int(inpt)
model_list = {"a":"F:/Work Folder/ML/best/test1.h5",
                          "b":"F:/Work Folder/ML/best/test4.h5",
                          "c":"F:/Work Folder/ML/best/test6.h5",
                          "d":"F:/Work Folder/ML/best/test7-1.h5",
                          "e":"F:/Work Folder/ML/best/test7-2.h5",
                          "f":"F:/Work Folder/ML/best/test7XXX.h5",
                          "g":"F:/Desktop/testF7.h5",
                          "h":"F:/Desktop/testF8.h5",
                          "i":"F:/Desktop/testF9.h5"}
              
face_cascade = cv2.CascadeClassifier('D:/Programs/python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') #path to haarcascade_frontalface_default stored on your local disk (part of opencv lib)
true_val= ["umji","eunha","yerin","sowon","sowon","sowon","sowon","yuju","yuju","sowon","umji","yerin","yerin","eunha","eunha","eunha","sowon","yuju","yuju","yerin","yuju","yuju","eunha","eunha","umji","umji","yerin","umji","sinb","sinb","sinb","yuju","sinb","eunha","umji","yerin","sinb","eunha","umji","sinb","sinb","yerin","umji","sowon","sowon","sinb","umji" ]
cap = cv2.VideoCapture(0)

def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = np.array(sct.grab(monitor))
        # Convert to PIL/Pillow Image
        sct_img = np.flip(sct_img[:, :, :3], 2)  # 1
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2RGB)  # 2
        return sct_img

def pred():

    for (i,(x,y,w,h)) in enumerate (faces):


        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
        
        roi_gray = np.array(roi_gray)
        roi_gray = cv2.resize(roi_gray,(40,40))
        #add dims and axis (otherwise tensorflow error)
        roi_gray = np.expand_dims(roi_gray, axis = 0)
        roi_gray = roi_gray[:, :, :, np.newaxis]
        #predict the result
        result = recognizer.predict(roi_gray)

        #get class of the prediction 
        result_class = str(result.argmax(axis=1))
        #txt= str(result_class)
        percent = str(int(np.nanmax(result))*100)
        #percents = str(percent*100)

        dict = {"[0]":"eunha","[1]":"sinb","[2]":"sowon","[3]":"umji","[4]":"yerin","[5]":"yuju"}
        #convert predicted class to string and compare to the  dict above
        #s = np.array2string(result_class) 
      
        member = (dict[result_class])   
        print ("Prediction: "+member+" "+percent+"%")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,65,152),2)
        cv2.putText(frame, member+percent+"%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #display windows and resize 
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("screen",1920,1080)
        cv2.imshow("screen", frame)
                            
        cv2.waitKey(1)
        
        
    
def summary():
        print ("list: from true val")
        print (true_val)
        print("")
        a = (np.array(true_val) == np.array(lista))
        s = a.tolist()
        print ("list: from prediction")
        print (s)
        ct = s.count(False)
        count = str(ct)
        num = int(ct)
        res = 47-num
        resss = str(res)
        rest = int(res)
        restr = str(rest)
        print("")
        print("Moel Summary"+" of " +final)
        print ("WRONG = "+count+"   "+"CORRECT ="+resss)
        perf = (rest/47)*100
        perff = str(perf)
        print ("performance:"+perff+"%")


        
def onscreen():
    global frame,faces
    while True:
        frame = capture_screenshot()
        #img_np = np.array(img)
        #frame = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(40,40))
        pred()
    cv2.destroyAllWindows
      


def webcam():
    global frame,faces
    while True:
        ret,frame = cap.read()
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(40,40))
        pred()
    cv2.destroyAllWindows


    
def auto():  
    image = [cv2.imread(file) for file in glob.glob("F:/Work Folder/ML/test_faces/test/face/*.jpg")]
    for images in image:
        faces = face_cascade.detectMultiScale(images,scaleFactor=1.3,minNeighbors=5,minSize=(40,40))
        for (i,(x,y,w,h)) in enumerate (faces):
            cv2.rectangle(images,(x,y),(x+w,y+h),(255,0,0),2)
            roi = images[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
            roi_gray = np.array(roi_gray)
            roi_gray = cv2.resize(roi_gray,(40,40))
            roi_gray = np.expand_dims(roi_gray, axis = 0)
            roi_gray = roi_gray[:, :, :, np.newaxis]
            #predict the result
            result = recognizer.predict(roi_gray)   
            result_class = result.argmax(axis=1)          
            dict = {"[0]":"eunha","[1]":"sinb","[2]":"sowon","[3]":"umji","[4]":"yerin","[5]":"yuju"}
            s = np.array2string(result_class) 
            member = (dict[s])        
            lista.append(member)
            counts = len(lista)
            count = str(counts)
            print (count+"/47")
            if count == 47:
                print ("list: from ML ")
                print (lista)
                print ("")     
    cv2.destroyAllWindows


#after select mode 
if inpt ==1:
    while inpt ==1:
        inputt = input("choose model: ")
        inputs = str(inputt)
        selected = (model_list[inputs])
        final = (os.path.relpath(selected, "F:/Work Folder/ML/best/"))
        print ("selected model: "+ final)
        print ("")
        recognizer = load_model(selected)
        recognizer.summary()
        lista=[]
        auto()     
        summary()
        
elif inpt ==2:
    inputt = input("choose model: ")
    inputs = str(inputt)
    selected = (model_list[inputs])
    final = (os.path.relpath(selected, "F:/Work Folder/ML/best/"))
    print ("selected model: "+ final)
    print ("")
    recognizer = load_model(selected)
    recognizer.summary()
    webcam()

elif inpt ==0:
    inputt = input("choose model: ")
    inputs = str(inputt)
    selected = (model_list[inputs])
    final = (os.path.relpath(selected, "F:/Work Folder/ML/best/"))
    print ("selected model: "+ final)
    print ("")
    recognizer = load_model(selected)
    recognizer.summary()
    onscreen()



