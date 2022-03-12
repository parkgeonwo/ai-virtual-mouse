import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)

# mediapipe hands 사용하기
mpHands = mp.solutions.hands
hands = mpHands.Hands()         
mpDraw = mp.solutions.drawing_utils

# fps를 나타내기위한 pTime, cTime 지정
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)      # none,none,none,,,,,,

    if results.multi_hand_landmarks:           # results.multi_hand_landmarks 가 존재하면
        for handLms in results.multi_hand_landmarks:    
            for id, lm in enumerate(handLms.landmark):     # 점들의 숫자를 알려주기위한 id, lm
                # print(id,lm)      # id(0부터 20) , x, y, z 가 0~1사이로 나타남
                h, w, c = img.shape     # height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h)  # find position, lm의 x value에 width 곱하기, y value에 height 곱하기
                # print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)    # id가 0인 지점에 원을 그려주기
                # cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED) # 모든 점에 원 그리기

            # mpDraw.draw_landmarks(img, handLms)      # img에 나타나는 손에 handLms를 점으로 표시해줌
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)    # 점들을 선으로 연결해서 나타냄

    # fps 계산
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText( img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )

    cv2.imshow("Image",img)

    if cv2.waitKey(10) == ord('q'):    # 프레임 넘어가는 속도 10ms( 프레임당 10ms 만큼 대기하면서 보여줌 ) / q를 누르면 꺼지도록 설정
        break







