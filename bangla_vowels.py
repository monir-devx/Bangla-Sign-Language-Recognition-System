import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from keys import *
import main as ma

class bangla_vowel:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the video file path.
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.banglaKey = Key(50, 5, 300, 50, '3.Bangla Vowels Activated')
        self.backKey = Key(150, 5, 150, 50, '4.Exit')
        self.frameHeight, self.frameWidth, _ = self.cam.read()[1].shape
        self.banglaKey.x = int(self.frameWidth * .73) - 150
        self.backKey.x = int(self.frameWidth * 1.4) - 150

    def is_a(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_pinky_mcp = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_pinky_mcp.x - right_index_tip.x) + abs(left_pinky_mcp.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.1

    def is_aa(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_thumb_tip = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_thumb_tip.x - right_index_tip.x) + abs(left_thumb_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def is_e(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_pinky_tip = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_pinky_tip.x - right_index_tip.x) + abs(left_pinky_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def is_u(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_middle_tip = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_middle_tip.x - right_index_tip.x) + abs(left_middle_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def is_ae(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_ring_tip = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_ring_tip.x - right_index_tip.x) + abs(left_ring_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def is_o(self, left_hand_landmarks, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        left_index_tip = left_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(left_index_tip.x - right_index_tip.x) + abs(left_index_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def is_back(self, right_hand_landmarks):
        # Check if the index finger tips of left and right hands are touching.
        if not right_hand_landmarks:
            return False

        right_thumb_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        right_index_tip = right_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        distance = abs(right_thumb_tip.x - right_index_tip.x) + abs(right_thumb_tip.y - right_index_tip.y)
        # Tweak the threshold value as per your requirement.
        return distance < 0.05

    def detect_gestures(self, imgg):
        imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(imgRGB)

        left_hand_landmarks = None
        right_hand_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x < 0.5:
                    left_hand_landmarks = hand_landmarks
                else:
                    right_hand_landmarks = hand_landmarks

                # Draw hand landmarks on the frame.
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * imgg.shape[1]), int(landmark.y * imgg.shape[0])
                    cv2.circle(imgg, (x, y), 5, (0, 255, 0), -1)

        return left_hand_landmarks, right_hand_landmarks


    def vowel_frame(self):

        while self.cam.isOpened():
            success, img = self.cam.read()

            if not success:
                break
            img = cv2.resize(img, (int(self.frameWidth * 1.5), int(self.frameHeight * 1.5)))

            imgg = cv2.flip(img, 1)
            self.banglaKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)
            self.backKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)

            left_hand_landmarks, right_hand_landmarks = self.detect_gestures(imgg)

            # Detect and display "Thumbs Up!" text if the index tips are touching.
            if self.is_a(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "অ", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_aa(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "আ", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_e(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "ই", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_u(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "উ", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_ae(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "এ", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_o(left_hand_landmarks, right_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fontpath = "fonts/Kalpurush-Regular.ttf"
                font = ImageFont.truetype(fontpath, 60)
                img_pil = Image.fromarray(imgg)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "ও", font=font, fill=(0, 0, 255, 0))
                imgg = np.array(img_pil)

            if self.is_back(left_hand_landmarks):
                #cv2.putText(imgg, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                exit()

            cv2.imshow('Hand Pose Detection', imgg)

            cv2.waitKey(1) # Press 'Esc' to exit the loop.
