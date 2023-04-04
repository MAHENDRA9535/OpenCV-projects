import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'madonna', 'mindy kaling']
DIR = r'C:\all files\projects\python\openCV\projects\Faces\train'

haar_cascade = cv.CascadeClassifier(
    'projects/haarcascade_frontalface_default.xml')
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            grey = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            face_react = haar_cascade.detectMultiScale(grey, 1.1, 4)

            for (x, y, z, w) in face_react:
                faces_roi = grey[y:y+w, x:x+z]
                features.append(faces_roi)
                labels.append(label)


create_train()

print("-----------training data------------")
features = np.array(features, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
