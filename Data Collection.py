#It is a independent file only for image-data collection.
#Key 'A' = Start & Stop 'N' = Next. Supports A to Z (26) files but extendable using number-fileName.
import cv2
import mediapipe as mp
import numpy as np
import math
import os as oss

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.imgSize = 300
        self.imgSize2 = 400
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.count = 0
        self.c_dir = 'A'
        self.offset = 15
        self.step = 1
        self.flag = False
        self.suv = 0

    def create_white_canvas(self, image):
        return 255 * np.ones_like(image)

    def track_hands(self):
        cap = cv2.VideoCapture(0)  # Change 0 to your video source if not using a webcam

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            # Convert the frame to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

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

                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(cropped_frame, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    #print(hCal+hGap)

                # Display the cropped image in a separate window
                cv2.imshow('Cropped Hand Landmarks', cropped_frame)
                cv2.imshow('White image', imgWhite)

            frame = cv2.putText(frame, "dir=" + str(self.c_dir) + "  count=" + str(self.count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hand', frame)

            interrupt = cv2.waitKey(1)
            if interrupt & 0xFF == 27:  # esc key
                break

            if interrupt & 0xFF == ord('n'):
                self.c_dir = chr(ord(self.c_dir) + 1)
                if ord(self.c_dir) == ord('Z') + 1:
                    self.c_dir = 'A'
                self.flag = False
                self.count = len(oss.listdir("C:\\Users\\asusd\\PycharmProjects\\Sign-Language-Communication-and-Calculation\\New folder\\" + self.c_dir + "\\"))

            if interrupt & 0xFF == ord('a'):
                if self.flag:
                    self.flag = False
                else:
                    self.suv = 0
                    self.flag = True

            print("=====", self.flag)
            if self.flag:
                if self.suv == 180:
                    self.flag = False
                if self.step % 3 == 0:
                    cv2.imwrite("C:\\Users\\asusd\\PycharmProjects\\Sign-Language-Communication-and-Calculation\\New folder\\" + self.c_dir + "\\" + str(self.count) + ".jpg",
                        imgWhite)
                    self.count += 1
                    self.suv += 1
                self.step += 1

        cap.release()
        cv2.destroyAllWindows()

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

if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.track_hands()
