import cv2
import face_recognition
import numpy as np
from trans_app import show_img
from trans_app2 import find_face_part, paint_face_part



original_face_image = 'images/face_img.png'
original_eye_image = 'images/eye_img.png'

def caluculate_left_eye_posi():
    left_eye_landmarks = find_face_part(face_part_name='left_eye')

    left_eye_top = min(left_eye_landmarks, key=lambda x: x[1])[1]
    left_eye_bottom = max(left_eye_landmarks, key=lambda x: x[1])[1]
    left_eye_left = min(left_eye_landmarks, key=lambda x: x[0])[0]
    left_eye_right = max(left_eye_landmarks, key=lambda x: x[0])[0]

    left_eye_height = int(left_eye_bottom) - int(left_eye_top)
    left_eye_width = int(left_eye_right) - int(left_eye_left)

    left_eye_posi = {
        'l_width': left_eye_width,
        'l_height': left_eye_height,
        'l_left': left_eye_left,
        'l_top': left_eye_top
    }

    return left_eye_posi


def caluculate_right_eye_posi():
    right_eye_landmarks = find_face_part(face_part_name='right_eye')

    right_eye_top = min(right_eye_landmarks, key=lambda x: x[1])[1]
    right_eye_bottom = max(right_eye_landmarks, key=lambda x: x[1])[1]
    right_eye_left = min(right_eye_landmarks, key=lambda x: x[0])[0]
    right_eye_right = max(right_eye_landmarks, key=lambda x: x[0])[0]

    right_eye_height = int(right_eye_bottom) - int(right_eye_top)
    right_eye_width = int(right_eye_right) - int(right_eye_left)

    right_eye_posi = {
        'r_width': right_eye_width,
        'r_height': right_eye_height,
        'r_left': right_eye_left,
        'r_top': right_eye_top
    }

    return right_eye_posi

left_eye_posi = caluculate_left_eye_posi()
right_eye_posi = caluculate_right_eye_posi()

# print(left_eye_posi)
# print(right_eye_posi)

def paste_mask(new_l_data, new_r_data, face_img, mask_img):

    lw = new_l_data['l_width']
    lh = new_l_data['l_height']
    l_top = new_l_data['l_top']
    l_left = new_l_data['l_left']

    rw = new_r_data['r_width']
    rh = new_r_data['r_height']
    r_top = new_r_data['r_top']
    r_left = new_r_data['r_left']

    resized_l_eye_img = cv2.resize(mask_img, (lw, lh))
    resized_r_eye_img = cv2.resize(mask_img, (rw, rh))

    change_l_height_end = l_top + lh
    change_l_width_end = l_left + lw

    change_r_height_end = r_top + rh
    change_r_width_end = r_left + rw

    resized_l_bgr = resized_l_eye_img[:, :, :3]
    resized_r_bgr = resized_r_eye_img[:, :, :3]

    l_eye_img_bgr = face_img[l_top:change_l_height_end, l_left:change_l_width_end]
    r_eye_img_bgr = face_img[l_top:change_r_height_end, l_left:change_r_width_end]

    resized_l_alpha = resized_l_eye_img[:, :, 3:] / 255
    resized_r_alpha = resized_r_eye_img[:, :, 3:] / 255

    resized_l_bgra = resized_l_bgr * resized_l_alpha
    resized_r_bgra = resized_r_bgr * resized_r_alpha

    l_img_bgra = l_eye_img_bgr * (1 - resized_l_alpha)
    r_img_bgra = r_eye_img_bgr * (1 - resized_r_alpha)

    mask = l_img_bgra + r_img_bgra + resized_l_bgra + resized_r_bgra

    face_img[l_top:change_l_height_end, l_left:change_l_width_end] = resized_l_eye_img
    face_img[r_top:change_r_height_end, r_left:change_r_width_end] = resized_r_eye_img

    face_img[l_top:change_l_height_end, l_left:change_l_width_end] = mask
    face_img[r_top:change_r_height_end, r_left:change_r_width_end] = mask

    return face_img

def main():
    left_eye_posi = caluculate_left_eye_posi()
    right_eye_posi = caluculate_right_eye_posi()

    face_img = cv2.imread(original_face_image)
    mask_img = cv2.imread(original_eye_image, cv2.IMREAD_UNCHANGED)
    face_img = paste_mask(left_eye_posi,right_eye_posi, face_img, mask_img)
    show_img(face_img)