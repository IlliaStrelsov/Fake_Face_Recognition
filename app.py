import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from keras.models import load_model
import pickle
from keras.utils.image_utils import img_to_array
from Attendance import Attendance
from werkzeug.security import generate_password_hash, check_password_hash
import Database.Database as database
from flask_session import Session

attendance = Attendance()
app = Flask(__name__)
fake_face_detector = load_model('FakeFaceIdentifier.model')
le = pickle.loads(open('le.pickle', "rb").read())
try:
    cap = cv2.VideoCapture(-1)
except:
    cap = cv2.VideoCapture(0)

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


@app.route('/home')
def home():
    names, times, l = attendance.extract_attendance()
    return render_template('home.html', names=names, times=times, l=l, totalreg=attendance.totalreg(),
                           datetoday2=attendance.datetoday2)


@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=attendance.totalreg(), datetoday2=attendance.datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            try:
                cap_new = cv2.VideoCapture(-1)
            except:
                cap_new = cv2.VideoCapture(0)
            ret, frame = cap_new.read()
        if attendance.extract_faces(frame) != ():
            if isinstance(attendance.extract_faces(frame), list):
                break
            (x, y, w, h) = attendance.extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = attendance.identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            preds = fake_face_detector.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            print(label)
            if label == 'fake':
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('fake'))
            elif label == 'real':
                attendance.add_attendance(identified_person)
                cap.release()
                cv2.destroyAllWindows()
                break
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    names, times, l = attendance.extract_attendance()
    return render_template('home.html', names=names, times=times, l=l, totalreg=attendance.totalreg(),
                           datetoday2=attendance.datetoday2)


@app.route('/fake')
def fake():
    return render_template('fake_page.html')


@app.route('/')
def signup():
    return render_template('signup.html')


@app.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('name')
    password = request.form.get('password')
    cursor = database.Database().getConnection().cursor()
    cursor.execute("SELECT * FROM users WHERE name='" + username + "'")
    users = cursor.fetchall()
    cursor.close()

    if users:
        flash('User address already exists')
        return redirect(url_for('signup'))

    cursor = database.Database().getConnection().cursor()
    cursor.execute("INSERT INTO users (name, password) VALUES('" + username + "','" + generate_password_hash(password,
                                                                                                             method='sha256') + "')")
    database.Database().getConnection().commit()
    cursor.close()

    cursor = database.Database().getConnection().cursor()
    cursor.execute("SELECT id FROM users WHERE name='" + username + "'")
    user_id = cursor.fetchall()[0][0]
    cursor.close()

    userimagefolder = 'static/faces/' + username + '_' + str(user_id)

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = attendance.extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = username + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Тренуємо модель')
    attendance.train_model()

    names, times, l = attendance.extract_attendance()
    return render_template('home.html', names=names, times=times, l=l, totalreg=attendance.totalreg(),
                           datetoday2=attendance.datetoday2)


if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.urandom(12).hex()
    app.config['SESSION_TYPE'] = 'filesystem'

    sess = Session()
    app.run(debug=True)
