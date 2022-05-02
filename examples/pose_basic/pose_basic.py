import cv2
import numpy as np
import time
import PoseModule as pm

# 웹캠으로 이미지 입력
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# pose detector 객체 생성
detector = pm.poseDetector()

# fps 계산을 위한 변수
pTime = 0

while True:
    # 비디오 촬영 시작
    success, img = cap.read()
    # 입력 이미지 크기 조절
    img = cv2.resize(img, (640, 480))

    # 이미지 내 pose 감지
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # 세 점의 랜드마크를 활용한 각도 구하기
        arm_angle = int(detector.findAngle(img, 12, 14, 16, draw=False))

        # 두 점의 랜드마크를 활용한 각도 구하기
        shoulder_angle = int(detector.findAngleTwoPoints(img, 12, 11, draw=False))

        # 두 점의 랜드마크 사이의 길이 구하기
        link_distance = int(detector.findDistance(img, 11, 13, draw=False))

        # 두 점의 랜드마크 사이의 중심 구하기
        mouth_point = detector.findCenterPoint(img, 10, 9, draw=False)
        center_shoulder_point = detector.findCenterPoint(img, 12, 11, draw=False)

        # 두 좌표 사이의 거리 구하기
        neck_distance = int(detector.findDistanceTwoPoints(img, mouth_point, center_shoulder_point, draw=False))
        
        print(f'arm_angle :{arm_angle:4d}\t shoulder_angle :{shoulder_angle:4d}\t link_distance :{link_distance:4d}\t neck_distance :{neck_distance:4d}')      

    # FPS 계산
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
