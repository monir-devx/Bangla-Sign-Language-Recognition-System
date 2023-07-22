import cv2
import mediapipe as mp
from keys import *
from PIL import ImageFont, ImageDraw, Image
import main as ma

class Communicator:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.cam = cv2.VideoCapture(0)
        self.x = []
        self.y = []
        self.text = ""
        self.k = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.idset = ["","1234","014", "234", "01234", "1", "3"]
        self.op = ["","আস-সালামু আলাইকুম","আিম েতামােক ভােলাবািস","ভােলা হেয়েছ"," েকমন আেছন ?","তুিম কী করেতেছা ?"]

        # copied from main.py to ensure same frame-size & show button
        self.communicationKey = Key(50, 5, 300, 50, '2.Communication')
        self.backKey = Key(150, 5, 150, 50, '3.Back')  # To add "back" button
        # getting frame's height and width
        self.frameHeight, self.frameWidth, _ = self.cam.read()[1].shape
        self.communicationKey.x = int(self.frameWidth * .73) - 150
        self.backKey.x = int(self.frameWidth * 1.4) - 150  # To add "back" button

    def communication_frame(self):
        while True:
            success, img = self.cam.read()

            # copied from main.py to ensure same frame-size
            if not success:
                break
            img = cv2.resize(img, (int(self.frameWidth * 1.5), int(self.frameHeight * 1.5)))

            imgg = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            # copied from main.py to show button
            self.communicationKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5)
            self.communicationKey.text = "2.communication Activated"
            # To add "back" button
            self.backKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5)
            self.backKey.text = "3.Back"

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = imgg.shape
                        if id == 0:
                            self.x = []
                            self.y = []
                        self.x.append(int((lm.x) * w))
                        self.y.append(int((1 - lm.y) * h))

                        if len(self.y) > 20:
                            id = ""
                            big = [self.x[3], self.y[8], self.y[12], self.y[16], self.y[20]]
                            small = [self.x[4], self.y[6], self.y[10], self.y[14], self.y[18]]

                            for i in range(len(big)):
                                if big[i] > small[i]:
                                    id += str(i)

                            try:
                                ind = self.idset.index(id)
                            except ValueError:
                                ind = 0
                            self.k[ind] += 1

                            for i in range(len(self.k)):
                                if self.k[i] > 20:
                                        if i == 6:
                                            main = ma.Main()
                                            main.process_frame()
                                        else:
                                          self.text = self.op[i]
                                          for i in range(len(self.k)):
                                            self.k[i] = 0

                    fontpath = "fonts/Kalpurush-Regular.ttf"
                    font = ImageFont.truetype(fontpath, 42)
                    img_pil = Image.fromarray(imgg)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((100, 100), self.text, font=font, fill=(0, 0, 255, 0))
                    imgg = np.array(img_pil)
                    self.mpDraw.draw_landmarks(imgg, handLms, self.mpHands.HAND_CONNECTIONS)
            else:
                self.text = " "

            cv2.imshow("WebCam", imgg)
            cv2.waitKey(1)
