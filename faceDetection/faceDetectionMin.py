import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
Ptime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)
    print(result)
    iimg = cv2.resize(img, (900,900))

    if result.detections:
        for id, detection in enumerate(result.detections):
            #mpDraw.draw_detection( iimg,detection)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = iimg.shape
            bbox = int(bboxC.xmin *iw), int(bboxC.ymin *ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(iimg,bbox, (255,0,255),2)
            cv2.putText(iimg, f'{int(detection.score[0]* 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1/cTime-Ptime
    Ptime = cTime
    cv2.putText(iimg,f'FPS:{int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow("image", iimg)
    cv2.waitKey(1)