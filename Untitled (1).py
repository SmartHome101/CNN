import numpy as np
import glob as gb
import cv2
import keras
import os
from collections import Counter




#load the model
KerasModel = keras.models.load_model('seven_mixed25.model')



#use open cv to det the user images
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/CNN/model/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
img_id = 0

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.8, minNeighbors=3)
  
    for face in faces:
        x,y,w,h=face
        
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        
        
        
        #save the frame images
        #change the name before run again
        file_name_path = "pre_test_cnn_input/"+"user"+str(img_id)+".jpg"
        cv2.imwrite(file_name_path, roi_color)
           
        
        y_end_cord=y+h
        x_end_cord=x+w
        color=(255,0,0)
        strock=2
        cv2.rectangle(frame,(x,y),(x_end_cord,y_end_cord),color,strock)
        img_id+=1
           
   

    if ret==True:
         cv2.imshow('frame1',frame)
    if cv2.waitKey(1)==13 or int(img_id)==50:
                break

        
cap.release()
cv2.destroyAllWindows()


#read the saved images and remove all images<140 pix then resize the remainig images

size=140
X_pred = []
files = gb.glob(pathname= str('pre_test_cnn_input//*.jpg'))
for file in files: 
    
    image = cv2.imread(file)
    
    #remove image less than 140 pix
    if image.shape[0]<140 and image.shape[1]<140:
        os.remove(file)  
        
    #resize the remainig images to 140 pix    
    else:    
        image_array = cv2.resize(image , (size,size))
    
        X_pred.append(list(image_array)) 

        
        
        
        

    

#use the model to predict how use the camera    
X_pred_array = np.array(X_pred)
#print(f'X_pred shape  is {X_pred_array.shape}')
y_result = KerasModel.predict(X_pred_array)




#set the ids
code = {'babars':0,'bassem':1,'elelfy':2,'eman':3,'ramdan':4,'roqia':5, 'sabek':6}
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x 


        

        
#get the user name
final_name=[]
for i in range(len(X_pred)): 
    
    final_name.append(getcode(np.argmax(y_result[i])))
#print(final_name) 
    

    

    
    
    
#get the real user name    
def most_common(final_name):
    data = Counter(final_name)
    return max(final_name, key=data.get)

the_user=most_common(final_name)
#printr the user name
print(the_user)    

