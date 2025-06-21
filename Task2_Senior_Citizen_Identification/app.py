
import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("age_gender_model.h5", compile=False)



gender_dict = {0: 'Male', 1: 'Female'}
output_file = "senior_log.csv"


if not hasattr(st.session_state, "init"):
    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=["Timestamp", "Age", "Gender", "SeniorCitizen"])
        df.to_csv(output_file, index=False)
    st.session_state.init = True

st.title("🎥 Real-Time Senior Citizen Detector")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)
    st.success("Webcam started. Looking for faces...")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not working.")
            break

     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (96, 96)) / 255.0  # Resizing to 96x96 as expected
            face_input = face_img.reshape(1, 96, 96, 1)         # Adding batch and channel dimensions


            pred_gender, pred_age = model.predict(face_input)
            age = int(round(pred_age[0][0]))
            gender = gender_dict[int(round(pred_gender[0][0]))]
            senior = "Yes" if age > 60 else "No"

       
            label = f"{gender}, Age: {age} ({'Senior' if senior == 'Yes' else 'Adult'})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

          
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_data = pd.DataFrame([[ts, age, gender, senior]], columns=["Timestamp", "Age", "Gender", "SeniorCitizen"])
            new_data.to_csv(output_file, mode='a', header=False, index=False)

       
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("Check the box to start webcam")
