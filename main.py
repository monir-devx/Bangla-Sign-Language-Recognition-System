import cv2
import mediapipe as mp
from keys import *
import calculation as cal
import communication as com
import bangla_alphabets as banal


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
        self.idset = ["", "1", "12","123", "4"]
        self.op = ["", "1", "2", "3"]

        # Creating keys
        self.w, self.h = 80, 60
        self.startX, self.startY = 40, 200

        self.calculationKey = Key(50, 5, 300, 50, '1.Calculation')
        self.communicationKey = Key(50, 60, 300, 50, '2.Communication')
        self.banglaKey = Key(50, 115, 300, 50, '3.Bangla Alphabets')
        self.exitKey = Key(150, 5, 150, 50, '4.Exit')  # To add "3.Exit" button

        # getting frame's height and width
        self.frameHeight, self.frameWidth, _ = self.cam.read()[1].shape
        self.calculationKey.x = int(self.frameWidth * .73) - 150
        self.communicationKey.x = int(self.frameWidth * .73) - 150
        self.banglaKey.x = int(self.frameWidth * .73) - 150
        self.exitKey.x = int(self.frameWidth * 1.4) - 150  # To add "3.Exit" button
        # print(showKey.x)

    def process_frame(self):
        while True:
            success, img = self.cam.read()

            if not success:
                break
            img = cv2.resize(img, (int(self.frameWidth * 1.5), int(self.frameHeight * 1.5)))

            imgg = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            self.calculationKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)
            self.communicationKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)
            self.banglaKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)
            # To add "3.Exit" button
            self.exitKey.drawKey(imgg, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.6)

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
                                    elif i == 2:
                                        communicator = com.Communicator()
                                        communicator.communication_frame()
                                    elif i == 3:
                                        bangla = banal.bangla_al()
                                        bangla.bangla_al_frame()
                                    elif i == 4:
                                        exit()
                                    else:
                                        self.text += self.op[i]
                                        for i in range(len(self.k)):
                                            self.k[i] = 0

                    self.mpDraw.draw_landmarks(imgg, handLms, self.mpHands.HAND_CONNECTIONS)
            else:
                self.text = " "

            cv2.imshow("WebCam", imgg)
            cv2.waitKey(1)


if __name__ == "__main__":
    main = Main()
    main.process_frame()
