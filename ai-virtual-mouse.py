import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#################

wCam, hCam = 640, 480

##############

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    # Find hand Landmarks
    success, img = cap.read()

    cv2.imshow("Image",img)
    if cv2.waitKey(10) == ord('q'):    # 프레임 넘어가는 속도 10ms( 프레임당 10ms 만큼 대기하면서 보여줌 ) / q를 누르면 꺼지도록 설정
        break



