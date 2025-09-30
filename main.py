import numpy as np
import face_recognition
import os, sys
import cv2
import PIL.Image
import PIL
from face_recognition import face_distance

# import image
imgElon = face_recognition.load_image_file('images/elon.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB) # change to RGB

imgTest = face_recognition.load_image_file('images/Elon_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB) # change to RGB


#ตรวจจับหน้า + ตีกรอบ (พิกัดหน้าบนภาพ)
faceLoc = face_recognition.face_locations(imgElon)[0] # [0] mean get firs element
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255), 3) #255,0,255 color:purple px:3

faceLoc_test = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc_test[3],faceLoc_test[0]), (faceLoc_test[1],faceLoc_test[2]), (255,0,255), 3) #255,0,255 color:purple px:3

results = face_recognition.compare_faces([encodeElon], encodeTest)
#compare_faces return True/False values indicating which known_face_encodings match the face encoding to check
faceDistance = face_recognition.face_distance([encodeElon], encodeTest) # lower distance better match
print(faceDistance, results) # elon vs elon ~ 0.41802574, elon vs lisa ~ 0.8

#img, text, org, fontFace: int, fontScale, color, thickness, lineType, bottomLeftOrigin
if(results):
    cv2.putText(imgTest, f'{"Elon:"}{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)



cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon test', imgTest)
cv2.waitKey(0)