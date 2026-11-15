# AI Vision Intelligence System

Hand gesture recognition using MediaPipe hand landmarks and a Random Forest classifier, with a Streamlit analytics dashboard.

## Project status and scope

- Five gesture classes: `fist`, `open_palm`, `one`, `two`, and `thumbs_up`.
- The committed raw landmark dataset contains 1,000 rows across five classes, with 200 samples per class.
- The engineered feature dataset contains the same 1,000 samples, represented by 63 normalized landmark features and one label column.
- The raw and engineered datasets have been checked row by row. Reapplying wrist-relative, scale normalization to the raw data reproduces the committed engineered values within a floating-point tolerance of 1e-7.
- `src/feature_engineering.py` can regenerate the engineered CSV from the committed raw CSV. It validates the input before writing the output.
- Automated CI checks validate dataset loading, shared inference preprocessing, raw-to-engineered dataset reproducibility, Python compilation, and a Streamlit dashboard smoke test.
- Automated tests do not verify physical webcam access or real-world gesture accuracy. The stored model artifact should be retrained and evaluated from the committed data before making a current performance claim.

## Gesture pipeline

1. Capture a hand image using a local webcam.
2. MediaPipe detects 21 hand landmarks.
3. Convert each landmark to wrist-relative coordinates and scale-normalize the 63 values.
4. Pass the feature row to the trained Random Forest model.
5. Display the predicted gesture and confidence in the webcam window.
6. Append timestamped predictions to `logs/predictions.csv`.
7. Review prediction history in the Streamlit dashboard.

The live inference path and training feature engineering use the same normalization definition. The shared live implementation is in `src/feature_utils.py`.

## Dataset

Each sample has 21 landmarks, each with X, Y, and Z coordinates, for 63 numerical features plus a `label` column.

| Property | Raw landmarks | Engineered features |
|---|---:|---:|
| Samples | 1,000 | 1,000 |
| Numerical features per sample | 63 | 63 |
| Gesture classes | 5 | 5 |
| Samples per class | 200 | 200 |

To regenerate the engineered dataset from the committed raw samples, run from the repository root:

```bash
python src/feature_engineering.py
```

The script validates the raw columns, labels, numeric values, and finite coordinates before replacing the engineered output.

## Machine learning

The project contains scripts to compare Logistic Regression, K-Nearest Neighbors, Random Forest, and Support Vector Machine models. The deployed live prediction script loads `models/gesture_random_forest.pkl`.

To train and save a Random Forest model from the current engineered dataset, run from the repository root:

```bash
python src/save_model.py
```

To print classification reports and confusion matrices for the four candidate models:

```bash
python src/evaluate_models.py
```

Do not treat the model's probability output as a calibrated measure of real-world correctness. A high confidence score is not a substitute for an independently measured test-set result. The checked-in model file has not yet been independently matched to a recorded training run and metric report.

## Setup

Use Python 3.11 in a virtual environment.

```bash
python -m venv venv
```

Windows activation:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Live webcam prediction (requires a working camera and the MediaPipe task asset):

```bash
python src/predict_live.py
```

Streamlit analytics dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard reads the committed prediction log. It does not itself access the webcam.

## Tests

Install test dependencies and run:

```bash
python -m pip install pandas numpy scikit-learn streamlit pytest
pytest -q tests/
```

GitHub Actions compiles the Python sources, runs the tests, and smoke-tests the Streamlit dashboard on relevant pull requests and pushes.

Latest successful CI run: [GitHub Actions run](https://github.com/krishankantpandey359-droid/AI-Vision-Intelligence/actions/runs/43727297001).

## Repository structure

```text
.
├── dashboard/
│   └── app.py
├── data/
│   ├── gesture_features.csv
│   └── gesture_landmarks.csv
├── logs/
│   └── predictions.csv
├── models/
│   ├── gesture_random_forest.pkl
│   └── hand_landmarker.task
├── src/
│   ├── dataset_utils.py
│   ├── feature_utils.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── save_model.py
│   ├── predict_live.py
│   └── camera_test.py
└── tests/
    └── test_dataset_utils.py
```

## Known limitations and next validation steps

- Retrain the checked-in Random Forest model from the committed engineered dataset and record a held-out classification report and confusion matrix.
- Run a real webcam session on a supported local machine. Hardware access is not available in GitHub-hosted CI.
- Validate the Streamlit deployment separately. A successful local smoke test does not prove the hosted URL is live.
- Review dependency compatibility for the local webcam stack before changing pinned package versions.
- Obtain an independent security and privacy review before collecting or publishing identifiable camera data.

## Author

Srishti Pandey, MCA Data Science
