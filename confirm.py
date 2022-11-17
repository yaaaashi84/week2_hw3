import cv2
import face_recognition
from trans_app import show_img

original_face_img = 'images/face_img.png'

face_image = face_recognition.load_image_file(original_face_img)
face_landmarks_list = face_recognition.face_landmarks(face_image)

print(face_landmarks_list)