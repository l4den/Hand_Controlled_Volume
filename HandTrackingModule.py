import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, complex = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complex = complex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        # convert to rgb image for the tracker
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        # if it detects hands
        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                if draw:
                    # draw the hand connections
                    self.mpDraw.draw_landmarks(img, handsLms, self.mpHands.HAND_CONNECTIONS)
        return img
                #for id, lm in enumerate(handsLms.landmark):
                #  h, w, c = img.shape  # video dimensions
                   # cx, cy = int(lm.x * w), int(lm.y * h)  # convert from decimals to pixels
                   # print(cx, cy)

    def findPosition(self, img, handNo=0, draw = True):

        lmList =[]
        if self.results.multi_hand_landmarks:
            #which hand
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape  # video dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # convert from decimals to pixels
                #print(cx, cy)
                lmList.append([id, cx, cy])

        return lmList



def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img)

        #if len(lmList)!=0:
            #print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # display fps                    where      font                 scale  color       thickness
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (83, 222, 68), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()