import cv2
import mediapipe as mp
import time 
import HandTrackingModule1 as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)        # img = detector.findHands(img, drwa=False) 하면 아무표시 안나옴 
    lmList = detector.findPosition(img)     # lmList = detector.findPosition(img, draw=False) 하면 점표시 없애줌
    # if len(lmList) != 0:
        # print(lmList[4])      # 4번째 점의 위치 출력

    # fps 계산
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText( img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )

    cv2.imshow("Image",img)

    if cv2.waitKey(1) == ord('q'):    # 프레임 넘어가는 속도 10ms( 프레임당 10ms 만큼 대기하면서 보여줌 ) / q를 누르면 꺼지도록 설정
        break








