import cv2
import numpy as np
import HandTrackingModule2 as htm
import time
import autopy

# 1. Find hand Landmarks
# 2. Get the tip of the index and middle fingers
# 3. Check which fingers are up
# 4. Only Index Finger : Moving Mode
# 5. Convert Coordicates 
# 6. Smoothen Values
# 7. Move Mouse
# 8. Both Index and middle fingers are up : Clicking Mode
# 9. Find distance between fingers
# 10. Click mouse if distance short
# 11. Frame Rate
# 12. Display

#################

wCam, hCam = 640, 480         # cam size
frameR = 30   # Frame Reduction
smoothening = 2      # 높을수록 속도가 느려짐

##############

pTime = 0             # Frame Rate 구하기 위한 value

plocX, plocY = 0, 0   # smoothening 하기 위한 value 
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()      # 내 스크린 사이즈 할당
# print(wScr, hScr, wScr/hScr)      # 1706.6666 , 1066.66666 , 1.6


while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]       # index finger = 검지 위치
        x2,y2 = lmList[12][1:]      # middle finger = 중지 위치
        # print(x1,y1,x2,y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), 
                            (255,0,255), 2 )   # 손가락 밖에 큰 네모 그려주기 , 이 네모안에서 전체적으로 움직일수있게       


        # 4. Only Index Finger : Moving Mode
        if fingers[1]==1 and fingers[2] ==0:      # 검지만 들고있을때
            
            # 5. Convert Coordicates 
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))   # 길이가 다른 두 1D 배열이 있을때, 길이를 맞춰줄 때 쓰는 np.interp
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

            # 6. Smoothen Values     # 이동속도를 부드럽게
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr-clocX,clocY)   # wScr-x3 을 하는 이유는 오른쪽으로 움직이면 오른쪽으로 움직이게 반전주기위해
            cv2.circle( img, (x1,y1), 15, (255,0,255), cv2.FILLED )

            plocX, plocY = clocX, clocY    # 업데이트

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1]==1 and fingers[2] ==1:       # 둘다 들고있을때만 작동
            
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8,12,img)       # 검지 중지 거리 구하기
            # print(length)    # 붙였을때 24-26 정도 나온다.
            # lineInfo 는 [x1, y1, x2, y2, cx, cy] 정보 담겨있다.

            # 10. Click mouse if distance short
            if length < 35:   # 두손가락이 어느정도 붙었을때 
                cv2.circle( img, ( lineInfo[4], lineInfo[5] ),   # cx, cy 지점 = 검지와 중지사이의 선의 중앙 지점
                        15, (0,255,0), cv2.FILLED ) # 검지와 중지사이의 선의 중앙 원의 색깔을 초록으로 변경
                autopy.mouse.click()     # 클릭
                time.sleep(0.5)


    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,0), 3 )

    # 12. Display
    cv2.imshow("Image",img)
    if cv2.waitKey(10) == ord('q'): # 프레임 넘어가는 속도 10ms( 프레임당 10ms 만큼 대기하면서 보여줌 ) / q를 누르면 꺼지도록 설정
        break



