import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import Database.Database as database
from datetime import date


class Attendance:

    datetoday = date.today().strftime("%m_%d_%y")
    datetoday2 = date.today().strftime("%d-%B-%Y")

    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def totalreg(self):
        return len(os.listdir('static/faces'))

    def extract_faces(self, img):
        if img != [] and img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = self.face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []

    def identify_face(self, facearray):
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)

    def train_model(self):
        faces = []
        labels = []
        userlist = os.listdir('static/faces')
        for user in userlist:
            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')

    def extract_attendance(self):
        cursor = database.Database().getConnection().cursor()
        cursor.execute("SELECT * FROM attendance")
        df = pd.DataFrame(cursor.fetchall())
        cursor.close()
        if df.empty:
            return [], [], 0
        names = df[1]
        times = df[2]
        l = len(df)
        return names, times, l


    def add_attendance(self, name):
        username = name.split('_')[0]
        cursor = database.Database().getConnection().cursor()
        cursor.execute("INSERT INTO attendance (user_name) VALUES('" + username + "')")
        database.Database().getConnection().commit()
        cursor.close()