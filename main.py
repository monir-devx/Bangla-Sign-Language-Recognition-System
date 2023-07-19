import cv2
import mediapipe as mp

class Calculator:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.cam = cv2.VideoCapture(0)
        self.x = []
        self.y = []
        self.text = ""
        self.k = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.idset = ["", "1", "12", "123", "1234", "01234", "0", "01", "012", "0123", "04", "4", "34", "014", "14", "234"]
        self.op = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/"]

    def process_frame(self):
        while True:
            success, img = self.cam.read()
            imgg = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

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
                                    if i == 15:
                                        ans = str(eval(self.text))
                                        self.text = self.text + " = " + ans
                                        for i in range(len(self.k)):
                                            self.k[i] = 0
                                    else:
                                        self.text += self.op[i]
                                        for i in range(len(self.k)):
                                            self.k[i] = 0

                    cv2.putText(imgg, self.text, (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 3)
                    self.mpDraw.draw_landmarks(imgg, handLms, self.mpHands.HAND_CONNECTIONS)
            else:
                self.text = " "

            cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("WebCam", 700, 700)
            cv2.imshow("WebCam", imgg)
            cv2.waitKey(1)

if __name__ == "__main__":
    calculator = Calculator()
    calculator.process_frame()
