import cv2
import mediapipe as mp
import time

#object for vid capture
cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands

hands = mpHands.Hands()

#for drawing points on the img
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

#frame capture
while True:
    success, img = cap.read()

    #convert to rgb image for the tracker
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    #if it detects hands
    if results.multi_hand_landmarks:
        for handsLms in results.multi_hand_landmarks:

            #pull point indexes and ther location
            for id, lm in enumerate(handsLms.landmark):

                h, w, c =img.shape #video dimensions
                cx, cy = int(lm.x*w), int(lm.y*h) #convert from decimals to pixels
                print(cx, cy)


            #draw the hand connections
            mpDraw.draw_landmarks(img, handsLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #display fps                    where      font                 scale  color       thickness
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (83, 222, 68), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)