
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import HandTrackingModule2 as htm
import time
import autopy
 

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
 

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # With Draw
    # hands = detector.findHands(img, draw=False)  # No Draw
 
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # Hand Type Left or Right
 
        # print(len(lmList1),lmList1)
        # print(bbox1)
        # print(centerPoint1)
        fingers1 = detector.fingersUp(hand1)
        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) # with draw
        #length, info = detector.findDistance(lmList1[8], lmList1[12])  # no draw
 
 
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left or Right
 
            fingers2 = detector.fingersUp(hand2)
            # print(fingers1, fingers2)
            print(lmList1[8][1:], lmList2[8][1:])
            length, info, img = detector.findDistance(lmList1[8][1:], lmList2[8][1:], img) # with draw
            #length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  # with draw
 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'): # 프레임 넘어가는 속도 1ms( 프레임당 1ms 만큼 대기하면서 보여줌 ) / q를 누르면 꺼지도록 설정
        break







