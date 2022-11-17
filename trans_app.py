import cv2
from PIL import Image

original_face_img = 'images/face_img.png'
original_eye_img = 'images/eye_img.png'

def find_face():
    cascade_file = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    img = cv2.imread(original_face_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_list = cascade.detectMultiScale(img_gray)

    return face_list

def paste_img(face_list):
    x = face_list[0][0]
    y = face_list[0][1]
    w = face_list[0][2]
    h = face_list[0][3]
    print(x, y, w, h)

    face_img = Image.open(original_face_img)
    eye_img = Image.open(original_eye_img)

    new_eye_img = eye_img.resize((w, h))
    new_eye_img.save('images/resized_eye_img.png')

    face_img.paste(new_eye_img, (x, y), new_eye_img.split()[3])

    face_img.save('images/pasted_face_img.png')

def show_img(face_img):
    cv2.imshow('face', face_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def main():
    face_list = find_face()
    paste_img(face_list)
    face_img = cv2.imread('images/pasted_face_img.png')
    show_img(face_img)

if __name__ == "__main__":
    main()
