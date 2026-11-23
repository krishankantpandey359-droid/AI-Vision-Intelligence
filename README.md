# AI Vision Intelligence System

Hand gesture recognition using MediaPipe hand landmarks and a Random Forest classifier, with a Streamlit analytics dashboard.

## Project status and scope

- Five gesture classes: `fist`, `open_palm`, `one`, `two`, and `thumbs_up`.
- The committed raw landmark dataset contains 1,000 rows across five classes, with 200 samples per class.
- The engineered feature dataset contains the same 1,000 samples, represented by 63 normalized landmark features and one label column.
- The raw and engineered datasets have been checked row by row. Reapplying wrist-relative, scale normalization to the raw data reproduces the committed engineered values within a floating-point tolerance of 1e-7.
- `src/feature_engineering.py` can regenerate the engineered CSV from the committed raw CSV. It validates the input before writing the output.
- Automated CI checks validate dataset loading, shared inference preprocessing, raw-to-engineered dataset reproducibility, Python compilation, and a Streamlit dashboard smoke test. CI evaluates the committed model before retraining, then evaluates the freshly trained model on the same deterministic stratified held-out split.
- The current CI workflow has evaluated both the committed model and a freshly trained Random Forest on the same deterministic stratified 20% holdout (200 samples). Both reports show 98.00% accuracy on this dataset split. These are held-out sample results, not independent-user, real-world or webcam accuracy. The original historical training provenance of the committed `.pkl` is still not established.

## Gesture pipeline

1. Capture a hand image using a local webcam.
2. MediaPipe detects 21 hand landmarks.
3. Convert each landmark to wrist-relative coordinates and scale-normalize the 63 values.
4. Pass the feature row to the trained Random Forest model.
5. Display the predicted gesture and confidence in the webcam window.
6. Append timestamped predictions to `logs/predictions.csv`. The logged percentage is the model probability for the displayed smoothed gesture.
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

This command also writes a held-out evaluation report to `outputs/retrained_model/metrics.json` and `outputs/retrained_model/confusion_matrix.csv`.

To compare Logistic Regression, K-Nearest Neighbors, Random Forest and Support Vector Machine with a reproducible held-out evaluation and five-fold stratified cross-validation:

```bash
python src/evaluate_models.py
```

The script creates `outputs/model_evaluation.json`. It records the dataset SHA-256 fingerprint, class counts, stratified 80/20 split, per-class reports, confusion matrices, and fold-level metrics with cross-validation means and standard deviations. Cross-validation runs only inside the 80% training partition. The held-out 20% is excluded from cross-validation and model selection.

This candidate-model comparison is separate from the committed-model check. CI first evaluates the checked-in `models/gesture_random_forest.pkl` with `src/evaluate_committed_model.py`, then runs the candidate comparison, and only afterwards retrains a fresh Random Forest with `src/save_model.py`. The committed model file is not replaced by the candidate comparison. CI uploads the candidate JSON report alongside the committed and retrained model reports.

Do not treat the model's probability output as a calibrated measure of real-world correctness. A high confidence score is not a substitute for independent testing. The checked-in model is evaluated by CI and its current report is recorded, but the historical training run that originally produced the committed `.pkl` has not been documented. See `MODEL_CARD.md` for the exact split and limitations. See [`docs/INDEPENDENT_EVALUATION_PROTOCOL.md`](docs/INDEPENDENT_EVALUATION_PROTOCOL.md) for the required group metadata and a participant/session-independent validation plan. The current dataset has no group identifiers, so a group-aware score is not yet available.

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

Collect new gesture samples (requires a working camera and the MediaPipe task asset). Existing raw data is retained and new rows are appended:

```bash
python src/collect_data.py
```

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

GitHub Actions evaluates the committed model first, runs the tests, retrains a fresh Random Forest, saves separate held-out metrics and confusion matrices for both models, and smoke-tests the Streamlit dashboard. The retrained model is a CI-runner artifact and is not automatically committed.

Latest successful CI run: [held-out evaluation workflow](https://github.com/krishankantpandey359-droid/AI-Vision-Intelligence/actions/runs/44596649004). Download the `ai-vision-heldout-evaluation` artifact from that run to inspect the committed-model and freshly retrained-model metrics and confusion matrices.

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

- Record the historical training command, training data version and model hash whenever the committed `.pkl` is regenerated. The original provenance of the currently committed artifact remains unknown.
- Run a real webcam session on a supported local machine. Hardware access is not available in GitHub-hosted CI. Data collection and live prediction resolve project asset paths relative to the script, so they do not depend on the terminal's current directory.
- Validate the Streamlit deployment separately. A successful local smoke test does not prove the hosted URL is live.
- Review dependency compatibility for the local webcam stack before changing pinned package versions.
- Obtain an independent security and privacy review before collecting or publishing identifiable camera data.

## Author

Srishti Pandey, MCA Data Science
