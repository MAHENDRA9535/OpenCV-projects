import cv2 as cv


cap = cv.VideoCapture("videos/2.mp4")

object_detector = cv.createBackgroundSubtractorMOG2(history=100,varThreshold=70)


while True:
    success,img = cap.read()
    frame= cv.resize(img, (900,900))

    mask = object_detector.apply(frame)
    _,mask = cv.threshold(mask,254,255,cv.THRESH_BINARY)
    contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv.contourArea(cnt)
        if area > 200:
            #cv.drawContours(frame,[cnt], -1,(255,0,255),1)
            x, y,w,h = cv.boundingRect(cnt)

            cv.rectangle(frame,(x,y),(x+w, y+h),(255,0,255, 3))

            detections.append([x,y,w,h])


    cv.imshow("mask",mask)

    cv.imshow("image",frame)
    cv.waitKey(1)
