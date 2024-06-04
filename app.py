import streamlit as st
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the put_on_sticker function
def put_on_sticker(img_bgr):

    # Load dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces in the image
    dlib_rects = detector(img_bgr, 1)

    # List to hold landmarks
    list_landmarks = []

    for dlib_rect in dlib_rects:
        landmarks = predictor(img_bgr, dlib_rect)
        list_landmarks.append([(p.x, p.y) for p in landmarks.parts()])

    # 각도에 맞게 돌린 이미지를 다시 정사각형의 형태로 맞춰주는 함수
    def get_largest_bounding_square(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate the size of the new bounding box
        radians = np.deg2rad(angle)
        cos = np.abs(np.cos(radians))
        sin = np.abs(np.sin(radians))
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # The new dimension must be the larger of the two to make the box square
        dim = max(new_w, new_h)

        # Adjust rotation matrix to take into account translation
        M = cv2.getRotationMatrix2D(center, angle, 1)
        M[0, 2] += (dim - w) / 2
        M[1, 2] += (dim - h) / 2

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (dim, dim), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return rotated

    # Process each detected face
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[33][0]
        y = landmark[33][1]
        w = h = dlib_rect.width()

        # 2와 14번 위치를 찾는다
        dx = landmark[14][0] - landmark[2][0]
        dy = landmark[14][1] - landmark[2][1]
        angle = -np.degrees(np.arctan2(dy, dx))

        # 스티커 이미지를 불러온다
        sticker_path = 'cat-whiskers.png'
        img_sticker = cv2.imread(sticker_path)
        img_sticker = cv2.resize(img_sticker, (w, h))

        # 스티커 이미지를 돌리고
        # 정사각형의 형태로 맞춰준다
        img_sticker_rotated = get_largest_bounding_square(img_sticker, angle)

        # Calculate the refined coordinates
        refined_x = x - img_sticker_rotated.shape[1] // 2
        refined_y = y - img_sticker_rotated.shape[0] // 2

        # Ensure the coordinates are within image bounds
        refined_x = max(0, min(refined_x, img_bgr.shape[1] - img_sticker_rotated.shape[1]))
        refined_y = max(0, min(refined_y, img_bgr.shape[0] - img_sticker_rotated.shape[0]))

        # Place the sticker on the image
        sticker_area = img_bgr[refined_y:refined_y + img_sticker_rotated.shape[0], refined_x:refined_x + img_sticker_rotated.shape[1]]
        img_bgr[refined_y:refined_y + img_sticker_rotated.shape[0], refined_x:refined_x + img_sticker_rotated.shape[1]] = \
            np.where(img_sticker_rotated == 255, sticker_area, img_sticker_rotated).astype(np.uint8)

    return img_bgr

# Streamlit app
st.title("😺얼굴에 고양이 수염을 붙여보아요😺")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the uploaded file to an opencv image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Apply the sticker
    img_with_sticker = put_on_sticker(img)

    # Display the result
    st.image(img_with_sticker, caption='Processed Image', use_column_width=True)
