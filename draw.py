import cv2
import numpy as np
import mediapipe as mp
import os

brushThickness = 12
eraserThickness = 50

drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)  

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

imgCanvas = np.zeros((720, 1280, 3), np.uint8)


xp, yp = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 
    h, w, c = img.shape

    cv2.rectangle(img, (0, 0), (w, 125), (0, 0, 0), cv2.FILLED) 
    
    # Red Button
    cv2.rectangle(img, (0, 0), (300, 125), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, "RED", (100, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    # Green Button
    cv2.rectangle(img, (300, 0), (600, 125), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, "GREEN", (380, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Blue Button
    cv2.rectangle(img, (600, 0), (900, 125), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, "BLUE", (680, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Eraser Button
    cv2.rectangle(img, (900, 0), (w, 125), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, "ERASER", (1000, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                x1, y1 = lmList[8][1], lmList[8][2]   # Index
                x2, y2 = lmList[12][1], lmList[12][2] # Middle

                fingers = []
                # Index
                if lmList[8][2] < lmList[6][2]: fingers.append(1)
                else: fingers.append(0)
                # Middle
                if lmList[12][2] < lmList[10][2]: fingers.append(1)
                else: fingers.append(0)

                if fingers[0] and fingers[1]:
                    xp, yp = 0, 0 
                    if y1 < 125:
                        if 0 < x1 < 300:
                            drawColor = (0, 0, 255) # Red
                        elif 300 < x1 < 600:
                            drawColor = (0, 255, 0) # Green
                        elif 600 < x1 < 900:
                            drawColor = (255, 0, 0) # Blue
                        elif 900 < x1 < w:
                            drawColor = (0, 0, 0)   # Eraser

                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                if fingers[0] and not fingers[1]:
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                    xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) > 0:
        break

cap.release()
cv2.destroyAllWindows()