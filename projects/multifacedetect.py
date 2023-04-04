import cv2 as cv

img = cv.imread('projects/photos/group 2.jpg')

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('group photo', grey)

cascade = cv.CascadeClassifier("projects/haarcascade_frontalface_default.xml")


face_rect = cascade.detectMultiScale(grey, 1.1, 6)
print(f'the number of faces = {len(face_rect)}')
for x, y, z, w in face_rect:
    cv.rectangle(img, (x, y), (x+z, y+w), (0, 255, 0), thickness=2)

cv.imshow("detected faces ", img)

cv.waitKey(0)
