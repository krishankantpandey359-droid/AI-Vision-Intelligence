# AI Vision Intelligence System

An end-to-end AI and Machine Learning project for real-time hand gesture recognition using computer vision.

## 📌 Project Overview

The AI Vision Intelligence System detects hand gestures in real time using a webcam. It uses MediaPipe to detect 21 hand landmarks and a Machine Learning model to classify different hand gestures.

The system also records prediction results and provides an interactive Streamlit dashboard for analyzing gesture predictions and confidence levels.

## 🎯 Gesture Classes

The system recognizes five hand gestures:

- Fist
- Open Palm
- One
- Two
- Thumbs Up

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Joblib
- Matplotlib / Altair

## 🔄 Project Workflow

Webcam
↓
Hand Detection
↓
21 Hand Landmarks
↓
Feature Engineering
↓
Machine Learning Model
↓
Real-Time Gesture Prediction
↓
Prediction Logging
↓
Streamlit Dashboard

## 📊 Dataset

The system uses 21 hand landmarks.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

This produces 63 numerical features for each hand sample.

The collected dataset contains 1000 samples across five gesture classes.

## ⚙️ Feature Engineering

The landmark coordinates are transformed using:

- Wrist-relative coordinates
- Scale normalization
- Numerical feature extraction

This helps make the model less dependent on hand position and distance from the camera.

## 🤖 Machine Learning Models

Multiple classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Support Vector Machine

Random Forest was selected for the real-time prediction system.

## 📈 Real-Time Prediction

The trained model predicts gestures from live webcam input and displays:

- Predicted gesture
- Prediction confidence
- Hand landmarks

Predictions are also stored in a CSV log file.

## 📊 Streamlit Dashboard

The project includes an interactive dashboard that displays:

- Total predictions
- Average confidence
- Most detected gesture
- Latest gesture
- Gesture distribution
- Confidence analysis
- Recent predictions

## 📁 Project Structure

```text
AI Vision Intelligence/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── gesture_landmarks.csv
│   └── gesture_features.csv
│
├── logs/
│   └── predictions.csv
│
├── models/
│   └── gesture_random_forest.pkl
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── camera_test.py
│   ├── collect_data.py
│   ├── evaluate_models.py
│   ├── extract_landmarks.py
│   ├── feature_engineering.py
│   ├── predict_live.py
│   ├── save_model.py
│   ├── train_models.py
│   └── train_test_split.py
│
├── .gitignore
├── README.md
└── requirements.txt


🚀 How to Run
1. Create and activate virtual environment
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run real-time prediction
python .\src\predict_live.py

4. Run the Streamlit dashboard
streamlit run .\dashboard\app.py


💡 Key Features

Real-time hand detection
21-point hand landmark extraction
Feature engineering
Multiple ML model comparison
Random Forest classification
Confidence-based prediction
Prediction logging
Interactive analytics dashboard
Modular project structure

🔮 Future Improvements
Add more hand gestures
Improve low-confidence predictions
Add model performance visualization
Deploy the Streamlit application
Add automatic model retraining
Support multiple hands
Add gesture-based application controls

👩‍💻 Author

Srishti Pandey
MCA | Data Science

AI & Machine Learning Project
# AI-Vision-Intelligence
End-to-end AI and Machine Learning system for real-time hand gesture recognition using MediaPipe, Random Forest and Streamlit.
