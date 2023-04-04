import numpy as np
import cv2 as cv


haar_cascade = cv.CascadeClassifier(
    'projects/haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'madonna', 'mindy kaling']

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(
    'C:/all files/projects/python/openCV/projects/Faces/val/ben_afflek/2.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)


face_react = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, z, w) in face_react:
    faces_roi = gray[y:y+w, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label{label} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+z, y+w), (0, 255, 0), thickness=2)

cv.imshow('detected face', img)

cv.waitKey(0)
