# To Comment & Uncomment multi-line Ctrl + /
#  1. Install cvzone
#  2. Install tensorflow
# 3. Run this file only (see errors)
# 4. From error description, open "ClassificationModule.py"
#    or "C:\Users\asusd\AppData\Local\Programs\Python\Python310\lib\site-packages\cvzone\ClassificationModule.py".
# 5. Edit it's 27th line from "label_file = open(self.labels_path, "r")"
#    to "label_file = open(self.labels_path, encoding="utf-8")"
# 6. Run again

import cv2
import mediapipe as mp
import numpy as np
import math
import os as oss
from cvzone.ClassificationModule import Classifier
from PIL import ImageFont, ImageDraw, Image

import bangla_alphabets as banal
from keys import *

class bangla_vowel:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.offset = 20
        self.imgSize = 300
        self.imgSize2 = 400
        self.hands = self.mp_hands.Hands()#static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.offset = 15
        self.cap = cv2.VideoCapture(0)  # Change 0 to your video source if not using a webcam
        # copied from main.py to ensure same frame-size & show button
        self.optionKey = Key(50, 5, 300, 50, 'Press "A" to Back')
        # getting frame's height and width
        self.frameHeight, self.frameWidth, _ = self.cap.read()[1].shape
        self.optionKey.x = int(self.frameWidth * .73) - 150

    def create_white_canvas(self, image):
        return 255 * np.ones_like(image)

    def vowel_frame(self):
        #cap = cv2.VideoCapture(0)  # Change 0 to your video source if not using a webcam
        #with open("Model/labels.txt", encoding="utf-8") as file:
            #content = file.read()

        classifier = Classifier("Model/keras_model.h5","Model/labels.txt" )  #
        labels = ["অ", "আ", "ই/ঈ", "উ/ঊ", "এ", "ঐ", "ও", "ঔ", "ঋ"]  #

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            frame = cv2.resize(frame, (int(self.frameWidth * 1.5), int(self.frameHeight * 1.5)))

            # Convert the frame to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # copied from main.py to show button
            self.optionKey.drawKey(frame, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)

            if results.multi_hand_landmarks:
                white_canvas = self.create_white_canvas(frame)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on the frame
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,circle_radius=2))
                    self.mp_drawing.draw_landmarks(white_canvas, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

                # Combine the original frame with the white canvas
                frame2 = cv2.addWeighted(frame, 1, white_canvas, 1, 0)

                # Get the bounding box coordinates of the hand landmarks
                x_min, x_max, y_min, y_max = self.get_bounding_box(frame2, hand_landmarks)

                # Crop the region of interest (hand landmarks connections)
                cropped_frame = frame2[y_min:y_max, x_min:x_max]

                h = y_max - y_min
                w = x_max - x_min
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(cropped_frame, (wCal, self.imgSize))
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    #print(prediction, index)  # To Print Prediction rate

                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(cropped_frame, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    #print(hCal+hGap)
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Display the cropped image in a separate window
                #cv2.imshow('Cropped Hand Landmarks', cropped_frame)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 42)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), labels[index], font=font, fill=(0, 0, 255, 0))
                frame = np.array(img_pil)
                #cv2.imshow('White image', imgWhite)  # To Show white image with landmarks

            #frame = cv2.putText(frame, "dir=" + str(self.c_dir) + "  count=" + str(self.count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('WebCam', frame)

            if cv2.waitKey(33) == ord('a'):
                bangla = banal.bangla_al()
                bangla.bangla_al_frame()


    def get_bounding_box(self, frame, hand_landmarks):
        x = [landmark.x for landmark in hand_landmarks.landmark]
        y = [landmark.y for landmark in hand_landmarks.landmark]

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        height, width, _ = frame.shape
        x_min, x_max = int((x_min * width) - self.offset), int((x_max * width) + self.offset)
        y_min, y_max = int((y_min * height) - self.offset), int((y_max * height) + self.offset)

        # Ensure the bounding box stays within the frame dimensions
        x_min = max(x_min, 0)
        x_max = min(x_max, width)
        y_min = max(y_min, 0)
        y_max = min(y_max, height)

        return x_min, x_max, y_min, y_max

