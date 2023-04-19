import cv2
import mediapipe as mp
import time

mpdraw = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime =0


while(True):
    success, img = cap.read()
    iimg = cv2.resize(img,(900,900),interpolation= cv2.INTER_LINEAR)

    imgRGB = cv2.cvtColor(iimg,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpdraw.draw_landmarks(iimg, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c =img.shape
            print(id,lm)
            cx,cy = int(lm.x*w),int(lm.y*h)
            #cv2.circle(iimg,(cx,cy),10,(255,0,255),cv2.FILLED)



    cTime =time.time()
    fps =1/(cTime-pTime)
    pTime = cTime

    cv2.putText(iimg, str(int(fps)),(70,50), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),3)

    cv2.imshow("image", iimg)
    cv2.waitKey(10)
