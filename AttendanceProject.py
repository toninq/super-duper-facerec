import numpy as np
import face_recognition
import cv2
from face_recognition import face_distance

import os #for find image in list and amount of image
from datetime import datetime

path = 'imagesAttendance' # เก็บ pathเป็นโฟลเดอร์ชื่อ imageAttendance
images = [] # list of data
classNames = []
myList = os.listdir(path) # path = imagesAttendance, ได้ทุกรายชื่อมาเก็บในลิสต์
                          # print(myList) : ['jisoo.jpg', 'lisa.jpg', 'reso.jpg']

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}') # path / img_filename read เก็บใน currImg
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0]) # split .jpg ทิ้ง
print(classNames) # ['jisoo', 'lisa', 'reso']


def markAttendance(name): # name, time arrived, ผูก database
    with open('Attendances.csv', 'r+', encoding='utf-8') as f:
        my_data_list = f.readlines()
        nameList = []
        for line in my_data_list:
            enytry = line.strip().split(',')  # strip() กัน \n
            nameList.append(enytry[0])
        if name not in nameList:
            now = datetime.now() # get current time
            date_string = now.strftime("%d/%m/%Y %H:%M:%S") # strftime representing date and time using date
            f.writelines(f'\n{name}, {date_string}\n')
        print(my_data_list)

markAttendance('lisa')

def findEncodings(images): # ยัด list of img คืน list of encoded img
    encodeList = [] # สร้างลิสว่าง
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert 2 rgb
        encode = face_recognition.face_encodings(img)[0] # encode rbg img using index0
        encodeList.append(encode) # append to list
    return encodeList

encodeListKnown = findEncodings(images) # list of people [safe : encoded แล้ว]

print(len(encodeListKnown), 'Encoding complete')

#init webcam
webCam = cv2.VideoCapture(0) # our id : 0

while True: # รับภาพมา frame by frame
    success, img = webCam.read()
    imgSmall = cv2.resize(img, (0,0), fx=0.25, fy=0.25)     # (0,0) ดู scale factor, x=0.25 y=0.25 => imgSmall = imgOriginal*0.25 ทั้ง x และ y
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # webCam found many faces
    # find face location to send to face_encode
    facesCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # ส่ง encodeListKnown เทียบกับ encodeFace
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace) # return list -> เลือก lowest distance มาเป็น best match
        print(faceDistance)

        #find lowest elm to be best match
        matchIndex = np.argmin(faceDistance) # return index

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
             # print(faceLoc)(top, right, bottom, left)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # rescale พิกัดกลับมาไซส์เดิม
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3) # cv2.rectangle(image, pt1บนซ้าย, pt2ล่างขวา, color, thickness)
            cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to cancel
        break

""" faceLoc = face_recognition.face_locations(imgElon)[0] # [0] mean get firs element
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255), 3) #255,0,255 color:purple px:3

faceLoc_test = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc_test[3],faceLoc_test[0]), (faceLoc_test[1],faceLoc_test[2]), (255,0,255), 3) #255,0,255 color:purple px:3

results = face_recognition.compare_faces([encodeElon], encodeTest)
#compare_faces return True/False values indicating which known_face_encodings match the face encoding to check
faceDistance = face_recognition.face_distance([encodeElon], encodeTest) # lower distance better match
print(faceDistance, results) # elon vs elon ~ 0.41802574, elon vs lisa ~ 0.8 """