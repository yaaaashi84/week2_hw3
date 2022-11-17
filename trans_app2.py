import face_recognition
import cv2
import numpy as np
from trans_app import show_img

original_face_img = 'images/face_img.png'

def find_face_part(face_part_name):
    face_image = face_recognition.load_image_file(original_face_img)
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    face_part_name_landmarks = face_landmarks_list[0][face_part_name]

    return face_part_name_landmarks

face_img = cv2.imread(original_face_img)
face_part_name = 'right_eye'
face_part_name_landmarks = find_face_part(face_part_name)
print(face_part_name_landmarks)


def paint_face_part(face_part_name_landmarks, face_img):
    points = np.array(face_part_name_landmarks)
    cv2.fillConvexPoly(face_img, points, color=(0, 0, 0))

    return face_img


for face_part_name_landmark in face_part_name_landmarks:
    cv2.drawMarker(
        face_img,
        face_part_name_landmark,
        color=(255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        thickness=1,
    )

show_img(face_img)