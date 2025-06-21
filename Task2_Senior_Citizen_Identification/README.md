# Task 2: Senior Citizen Identification in Real-Time

## 📌 Objective
Detect multiple people in real-time via webcam or video feed, estimate their age and gender, and log if they are senior citizens (age > 60).

## 📂 Dataset Used
- **Name**: UTKFace – 20k+ cropped face images with age, gender and ethnicity
- **Source**: [https://www.kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Used a preprocessed subset of ~ 20k+  cleaned images to reduce memory usage and training time.


## 🧠 Model
- Custom CNN for Age and Gender prediction
- Gender: Classification (Male/Female)
- Age: Regression

## 📁 Files
- `model_training.ipynb`: CNN training pipeline for age and gender.
- `app.py`: Real-time detection and logging script.
- `age_gender_model.h5`: Contains the trained model .
- `senior_log.csv`: Stores [Age, Gender, Timestamp].
- `requirement.txt`: All dependencies.
- `README.md`: This file.

## 📊 Evaluation Metrics
- Accuracy (Gender): 86%
- MAE (Age): 4.8
- Precision/Recall (Senior class): 0.88 / 0.85


## ▶️ How to Run
```bash
pip install -r requirement.txt
python app.py
