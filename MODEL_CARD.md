# Model Card: Gesture Random Forest

## Model identity

- Repository model path: `models/gesture_random_forest.pkl`
- Model type: `sklearn.ensemble.RandomForestClassifier`
- Number of input features: 63
- Gesture classes: `fist`, `one`, `open_palm`, `thumbs_up`, `two`
- Training script for a reproducible replacement: `src/save_model.py`
- Training parameters in the current script: 200 trees, `random_state=42`
- Feature preparation: shared wrist-relative and scale-normalized MediaPipe landmark pipeline.

## Current CI evaluation

The original detailed two-model metrics below were produced by GitHub Actions run [44596649004](https://github.com/krishankantpandey359-droid/AI-Vision-Intelligence/actions/runs/44596649004). The latest post-merge validation run is [45308866069](https://github.com/krishankantpandey359-droid/AI-Vision-Intelligence/actions/runs/45308866069). Do not treat the newer run as a replacement for the historical metric artifact unless its reports are downloaded and independently compared. The original run evaluated:

1. The committed `models/gesture_random_forest.pkl` before the workflow retrained anything.
2. A fresh Random Forest trained by `src/save_model.py` during the same run.

Both reports used a stratified 80/20 split of the committed 1,000-row engineered dataset, with `random_state=42`. The test partition contained 200 rows, 40 per class.

| Metric | Committed model | CI-retrained model |
|---|---:|---:|
| Held-out accuracy | 98.00% | 98.00% |
| Test samples | 200 | 200 |
| Features | 63 | 63 |
| Macro F1 | 98.00% | 98.00% |

Both models produced the same confusion matrix on this split:

| Actual  Predicted | fist | one | open_palm | thumbs_up | two |
|---|---:|---:|---:|---:|---:|
| fist | 39 | 0 | 1 | 0 | 0 |
| one | 0 | 40 | 0 | 0 | 0 |
| open_palm | 1 | 0 | 38 | 0 | 1 |
| thumbs_up | 0 | 0 | 0 | 40 | 0 |
| two | 1 | 0 | 0 | 0 | 39 |

The machine-readable reports and confusion matrices are stored in the `ai-vision-heldout-evaluation` artifact for the linked run.

## Intended use

This model is an educational hand-gesture recognition project for the five listed classes. It is intended for experimentation with MediaPipe landmarks, feature engineering, classification and dashboard display.

## Important limitations

- The 98% result is measured on a random held-out partition of the same committed dataset. It is not an independent dataset or independent participant evaluation.
- The README and repository history do not establish the original training date, environment, code revision or exact training data version used to create the currently committed pickle file. CI evaluates the committed file as it exists, but that does not reconstruct its historical provenance.
- The identical committed/retrained scores and confusion matrices show agreement on this one split only. They do not prove the committed model was produced by the current training script.
- No real webcam test has been performed by GitHub Actions. Camera hardware access is outside the CI environment.
- Performance may change with lighting, camera angle, hand size, skin/background contrast, occlusion, motion, and users not represented in the dataset.
- Predicted probabilities must not be interpreted as calibrated real-world correctness probabilities.
- Do not use this educational classifier for safety-critical, identity, biometric or other high-stakes decisions.

## Reproducing the current training procedure

From the repository root, install the project dependencies and run:

```bash
python src/save_model.py
```

The script trains a Random Forest from the committed engineered dataset, writes a model file, and saves held-out metrics in `outputs/retrained_model/`. The script overwrites the local model path. Review the output, preserve the evaluation report, and verify changes before committing a replacement model.


## Independent evaluation status

A group-aware evaluation is not currently possible from the committed CSV schema. The raw and engineered files contain no participant, session, source-video, or capture-device identifiers. Do not infer groups from row order or claim the existing 98% random-row holdout is independent-user performance. See [`docs/INDEPENDENT_EVALUATION_PROTOCOL.md`](docs/INDEPENDENT_EVALUATION_PROTOCOL.md) for the metadata to collect and the recommended participant-held-out and session-held-out evaluation design.
