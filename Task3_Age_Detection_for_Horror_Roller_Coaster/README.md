# Task 3: Real-Time Age Detection for Horror Roller Coaster Entry System

## 📌 Objective
Restrict entry for people aged <13 or >60 using webcam-based real-time face detection and age/gender classification.

## 📂 Dataset Used
- **Name**: UTKFace – 20k+ cropped face images with age, gender and ethnicity
- **Source**: [https://www.kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Used a preprocessed subset of ~ 20k+  cleaned images to reduce memory usage and training time.

## 🧠 Model
- CNN for Age and Gender prediction
- Classification logic: “Not allowed” for age <13 or >60

## 📁 Files
- `model_training.ipynb`: CNN training pipeline for age and gender.
- `app.py`: Real-time detection and logging script.
- `age_gender_model.h5`: Contains the trained model .
- `roller_coaster_log.csv`: Stores [Age, Gender, Timestamp].
- `requirement.txt`: All dependencies.
- `README.md`: This file.


## 📊 Evaluation Metrics
- Age MAE: 5.1
- Gender Accuracy: 85%
- Entry Restriction Logic: 100% correct trigger


## 🚦 System Behavior
- Green box: Allowed
- Red box: “Not Allowed” overlay
- Entry details logged to `roller_coaster_log.csv`

## ▶️ How to Run
```bash
pip install -r requirement.txt
python app.py
