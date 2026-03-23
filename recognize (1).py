import urllib
import cv2
import numpy as np
from keras.models import load_model


import pandas as pd
from datetime import datetime
import os


classifier = cv2.CascadeClassifier(r'C:\Users\LENOVO\OneDrive\Desktop\FACE DETECTION\FACE DETECTION\haarcascade_frontalface_default (1).xml')

model = load_model(r"C:\Users\LENOVO\OneDrive\Desktop\FACE DETECTION\FACE DETECTION\recognization_model.h5")

URL = 'http://192.190.86.69:8080/shot.jpg'


def get_pred_label(pred):
    labels = ["SAURAJEET","AKHILESH","SRIKANT"]
    return labels[pred]


def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img



def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    file_name = (r"C:\Users\LENOVO\OneDrive\Desktop\Attendance.xlsx")

    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(file_name, index=False)

    df = pd.read_excel(file_name)

    if not ((df["Name"] == name) & (df["Date"] == date_string)).any():
        new_row = {"Name": name, "Date": date_string, "Time": time_string}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(file_name, index=False)
        print("Attendance Marked for", name)
    else:
        print("Attendance already marked today")


ret = True
predicted_name = None   

while ret:
    
    img_url = urllib.request.urlopen(URL)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
      
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        
        
        predicted_name = get_pred_label(np.argmax(model.predict(preprocess(face))))
        
        cv2.putText(frame,predicted_name,
                    (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),2)
        
    cv2.imshow("capture",frame)

    key = cv2.waitKey(1)

    
    if key == ord('o'):
        if predicted_name is not None:
            mark_attendance(predicted_name)
        else:
            print("No face detected")


    if key == ord('q'):
        break


cv2.destroyAllWindows()
