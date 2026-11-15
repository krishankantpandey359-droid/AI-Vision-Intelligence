# AI Vision Intelligence System

Hand gesture recognition using MediaPipe hand landmarks and a Random Forest classifier, with a Streamlit analytics dashboard.

## Project status and scope

- Five gesture classes: `fist`, `open_palm`, `one`, `two`, and `thumbs_up`.
- The engineered feature dataset currently contains 1,000 rows, 63 landmark features, and 200 rows per class.
- `data/gesture_landmarks.csv` is currently empty. The original raw data has not been recreated or replaced. Therefore, feature engineering cannot currently be rerun from the repository's raw input.
- The existing engineered feature CSV and trained model are retained. The model's training provenance and performance need to be re-established from raw samples before claiming a fully reproducible training pipeline.
- Automated CI checks validate the dataset loader, shared inference preprocessing, Python compilation, and a Streamlit dashboard smoke test.
- Automated tests do not verify physical webcam access or real-world gesture accuracy.

## Gesture pipeline

1. Capture a hand image using a local webcam.
2. MediaPipe detects 21 hand landmarks.
3. Convert each landmark to wrist-relative coordinates and scale-normalize the 63 values.
4. Pass the feature row to the trained Random Forest model.
5. Display the predicted gesture and confidence in the webcam window.
6. Append timestamped predictions to `logs/predictions.csv`.
7. Review prediction history in the Streamlit dashboard.

The live inference path and training feature engineering now use the same normalization definition. The shared implementation is in `src/feature_utils.py`.

## Dataset

Each sample has 21 landmarks, each with X, Y, and Z coordinates, for 63 numerical features plus a `label` column.

Current engineered dataset:

| Property | Current value |
|---|---:|
| Samples | 1,000 |
| Features per sample | 63 |
| Gesture classes | 5 |
| Samples per class | 200 |

The raw landmark file is empty at present. Do not run `src/feature_engineering.py` until new raw samples have been collected. The script intentionally refuses an empty input and leaves the existing engineered dataset unchanged.

## Machine learning

The project contains scripts to compare Logistic Regression, K-Nearest Neighbors, Random Forest, and Support Vector Machine models. The deployed live prediction script loads `models/gesture_random_forest.pkl`.

Do not treat the model's probability output as a calibrated measure of real-world correctness. A high confidence score is not a substitute for an independently measured test-set result.

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

GitHub Actions also compiles the Python sources, runs the tests, and smoke-tests the Streamlit dashboard on relevant pull requests and pushes.

Latest successful CI run for shared live preprocessing: [GitHub Actions run](https://github.com/krishankantpandey359-droid/AI-Vision-Intelligence/actions/runs/43725303095).

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

- Collect and preserve a new raw landmark dataset before rerunning feature engineering and model training.
- Re-train the model from the documented raw data and record a held-out classification report and confusion matrix.
- Run a real webcam session on a supported local machine. Hardware access is not available in GitHub-hosted CI.
- Validate the Streamlit deployment separately. A successful local smoke test does not prove the hosted URL is live.
- Review dependency compatibility for the local webcam stack before changing pinned package versions.
- Obtain an independent security and privacy review before collecting or publishing identifiable camera data.

## Author

Srishti Pandey, MCA Data Science
