#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2


# In[ ]:


import numpy as np


# In[ ]:


cap = cv2.VideoCapture(0) #if u have multiple camera then u can write 1 or 2

face_cascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

eye_cascade =cv2.CascadeClassifier("haarcascade_eye.xml")



# In[ ]:


while True:
    _, img =cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1 ,4)
    
    #dimension of the bounding box around the face
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w ,y+h),(0,255,0))
        cv2.putText(img,'Faria',(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),1)# text on the face box 
        
        
    eyes = eye_cascade.detectMultiScale(gray, 1.1 ,4)
    
    #dimension of the bounding box around the eyes 
    
    for(x1,y1,w1,h1) in eyes:
        cv2.rectangle(img,(x1,y1),(x1+w1 ,y1+h1),(0,255,0))
        cv2.putText(img,'Eye',(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),1) #text on the eye box
    
    
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




