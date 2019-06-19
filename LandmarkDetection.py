import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

def draw_landmarks(image):
    img = cv2.imread(image)
    plt.figure(figsize=(10,8))
    try: 
        faces = detector(img)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            landmarks = predictor(img,face)
            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img,(x,y),4,(255,0,0),1)
                #print(x,y)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
        plt.imshow(img)
        plt.show()
    except:
        print("")
        
root_dir = "/home/aman/BU/Intern/Project/FacialDataFrame/927_0198_01"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(root_dir):
    for file in f:
        files.append(os.path.join(r, file))
        
for file in files:
    draw_landmarks(file)
