import cv2
import mediapipe as mp
from keys import *
import calculation as cal
class Main:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.cam = cv2.VideoCapture(0)
        self.x = []
        self.y = []
        self.text = ""
        self.k = [0, 0, 0, 0]
        self.idset = ["", "1", "12","234"]
        self.op = ["", "1", "2"]

        # Creating keys
        self.w, self.h = 80, 60
        self.startX, self.startY = 40, 200

        self.calculationKey = Key(50, 5, 300, 50, '1.Calculation')
        self.communicationKey = Key(50, 70, 300, 50, '2.Communication')
        self.textBox = Key(self.startX, self.startY - self.h - 5, 10 * self.w + 9 * 5, self.h, '')

        # getting frame's height and width
        self.frameHeight, self.frameWidth, _ = self.cam.read()[1].shape
        self.calculationKey.x = int(self.frameWidth * .73) - 150
        self.communicationKey.x = int(self.frameWidth * .73) - 150
        # print(showKey.x)

        self.calculation = False
        self.Communication = False


    def process_frame(self):
        while True:
            success, img = self.cam.read()

            if not success:
                break
            img = cv2.resize(img, (int(self.frameWidth * 1.5), int(self.frameHeight * 1.5)))

            imgg = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            self.calculationKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5)
            self.communicationKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5)

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
                                    if i == 1:
                                        calculator = cal.Calculator()
                                        calculator.calculation_frame()
                                    #elif i == 2:
                                        # here, code for communication.py
                                    else:
                                        self.text += self.op[i]
                                        for i in range(len(self.k)):
                                            self.k[i] = 0

                    cv2.putText(imgg, self.text, (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 3)
                    self.mpDraw.draw_landmarks(imgg, handLms, self.mpHands.HAND_CONNECTIONS)
            else:
                self.text = " "

            cv2.imshow("WebCam", imgg)
            cv2.waitKey(1)

if __name__ == "__main__":
    main = Main()
    main.process_frame()
