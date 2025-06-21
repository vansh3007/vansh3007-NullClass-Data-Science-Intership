# 🎓 NullClass Machine Learning Internship – Vansh Shrivastava

## 📝 Overview

This repository contains the complete work done during my **Data Science Internship** at **NullClass** from **May 21, 2025 to June 21, 2025**. The internship focused on implementing custom deep learning models for **real-world age and gender detection applications**. Each task required creating models from scratch, ensuring high performance, and integrating real-time functionality.

---

## 📚 Tasks Completed

### ✅ Task 1: Age Detection on IMDB-WIKI
- Built a custom **CNN model** to predict age from facial images.
- Trained on a cleaned subset of the **IMDB-WIKI dataset**.
- Evaluated using **MAE** and custom accuracy (±5 years).
- 📁 [Task 1 Folder](./Task1_IMDB_WIKI_Age_Detection/)

### ✅ Task 2: Senior Citizen Identification (Real-Time)
- Real-time webcam-based detection of multiple faces.
- Predicted **age and gender**, flagged senior citizens (age > 60).
- Logged data (age, gender, timestamp) to **CSV file**.
- 📁 [Task 2 Folder](./Task2_Senior_Citizen_Identification/)

### ✅ Task 3: Age-Based Entry Restriction for Horror Roller Coaster
- Live age/gender detection from webcam feed.
- Blocked people **<13 or >60 years** from entry.
- Overlayed a “Not Allowed” warning with red bounding boxes.
- Logged all entries to **CSV**, same model reused with entry logic.
- 📁 [Task 3 Folder](./Task3_Age_Detection_for_Horror_Roller_Coaster/)

---

## 📂 Dataset Used

- **IMDB Wiki Faces Dataset**: Publicly available dataset of over 500k face images .
- Used preprocessed grayscale/resized images for faster training and real-time inference.
- Dataset Source: [IMDB-WIKI Dataset](https://www.kaggle.com/datasets/abhikjha/imdb-wiki-faces-dataset/)


- **UTKFace**: 20k+ cropped face images with age, gender and ethnicity.
- Used preprocessed grayscale/resized images for faster training and real-time inference.
- Dataset Source: [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new/)

---

## 📈 Performance Summary

| Task | Metric | Value |
|------|--------|-------|
| Task 1 | MAE | 5.2 |
| Task 2 | Gender Accuracy | 86% |
| Task 2 | Senior Detection Precision/Recall | 0.88 / 0.85 |
| Task 3 | Entry Restriction Logic Accuracy | 100% (Test Set) |

---

## 📁 Folder Structure

```
/
├── Task1_IMDB_WIKI_Age_Detection/
│   ├── model_training.ipynb
│   ├── age_model.h5
│   ├── app.py
│   ├── requirement.txt
│   └── README.md
├── Task2_Senior_Citizen_Identification/
│   ├── model_training.ipynb
│   ├── age_model.h5
│   ├── app.py
│   ├── senior_log.csv
│   ├── requirement.txt
│   └── README.md
├── Task3_Age_Detection_for_Horror_Roller_Coaster/
│   ├── model_training.ipynb
│   ├── age_model.h5
│   ├── app.py
│   ├── roller_coaster_log.csv
│   ├── requirement.txt
│   └── README.md
├── data/
│   ├── imdb_crop //Dataset folder
│   ├── UTKFace //Dataset folder
├── internship_report.pdf
└── README.md ← (you are here)
```

---

## 🧪 How to Run

Each task contains:
- `requirement.txt` file for dependencies.
- Training and inference files.
- Instructions in their respective README files.
- Download Datasets using provided link and unzip them in data folder.

To install dependencies for a task:
```bash
pip install -r requirement.txt
```

---

## 🎯 Key Takeaways

- Developed 3 independent ML systems using CNN models.
- Learned and applied concepts of image preprocessing, model tuning, and real-time video analysis.
- Gained hands-on experience in OpenCV, TensorFlow, and project packaging for deployment.

---


## 📬 Contact

**Vansh Shrivastava**  
Email: vanshshrivastava30@gmail.com  
GitHub: [github.com/vansh3007](https://github.com/vansh3007)

---