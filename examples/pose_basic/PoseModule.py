import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self,
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=True,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                    self.model_complexity,
                                    self.smooth_landmarks,
                                    self.enable_segmentation,
                                    self.smooth_segmentation,
                                    self.min_detection_confidence,
                                    self.min_tracking_confidence)

    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:                
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:        
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy,lm.z])
                if draw:
                    cv2.circle(img,(cx,cy), 8, (255,0,0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):

        # 랜드마크 좌표 얻기
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        # 각도 계산
        radian = math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2)
        angle = math.degrees(radian)

        if angle < 0:
            angle += 360

        # 점, 선 그리기
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(img, (x2,y2), (x3,y3), (255,255,255), 3)            
            cv2.circle(img, (x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 15, (0,0,255), 2)
            cv2.putText(img, str(int(angle)), (x2-50,y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2) 
            
        return angle

    def findAngleTwoPoints(self, img, p1, p2, draw=True):
        
        # 랜드마크 좌표 얻기
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]

        # 각도 계산
        radian = math.atan2(y2-y1,x2-x1)
        angle = math.degrees(radian)

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.circle(img, (x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.putText(img, str(int(angle)), (x2-50,y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2) 
        return angle

    def findDistance(self, img, p1, p2, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2-x1, y2-y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0,0,255), cv2.FILLED)

        return length

    def findDistanceTwoPoints(self, img, p1, p2, draw=True, r=15, t=3):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2-x1, y2-y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0,0,255), cv2.FILLED)

        return length

    def findCenterPoint(self, img, p1, p2, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (cx, cy), r, (255,0,255), cv2.FILLED)

        return (cx, cy)

def main():
    cap = cv2.VideoCapture('PoseVideos/3.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img,(lmList[14][1],lmList[14][2]), 15, (0,0,255), cv2.FILLED)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)

        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()