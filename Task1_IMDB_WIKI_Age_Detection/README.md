# Task 1: Age Detection on IMDB-WIKI Dataset

## 📌 Objective
Build a CNN model from scratch to predict the age of a person based on facial images using the IMDB-WIKI dataset.

## 📂 Dataset Used
- **Name**: IMDB Wiki Faces Dataset – 500k+ face images with age labels.
- **Source**: [https://www.kaggle.com/datasets/abhikjha/imdb-wiki-faces-dataset](https://www.kaggle.com/datasets/abhikjha/imdb-wiki-faces-dataset)
- Used a preprocessed subset of ~50,000 cleaned images to reduce memory usage and training time.


## 🧠 Model
- CNN with Conv2D, MaxPooling, Dropout, Dense layers
- Optimizer: Adam
- Loss: Mean Absolute Error (MAE)
- Achieved Accuracy: **72.3%**

## 📁 Files
- `model_training.ipynb`: Contains data preprocessing, model building, training, and evaluation.
- `age_model.h5`: Contains the trained model .
- `requirement.txt`: All dependencies required to run the code.
- `README.md`: This file.

## 📊 Evaluation Metrics
- MAE: 5.2
- Confusion matrix: NA (regression task)
- Custom accuracy calculated on ±5 years margin


## ▶️ How to Run
```bash
pip install -r requirement.txt
jupyter notebook model_training.ipynb
