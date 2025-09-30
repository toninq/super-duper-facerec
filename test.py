import face_recognition
import numpy as np

img = face_recognition.load_image_file('images/elon.jpg')
print(img.shape, img.dtype)
