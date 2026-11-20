# Independent Evaluation Protocol

## Current evidence

The repository contains 1,000 engineered landmark rows across five gesture classes. The current CI result uses a deterministic stratified random 80/20 split of those rows. It reports 98% accuracy for both the committed model and a newly trained Random Forest.

This is a row-level holdout only. The available CSV schema contains landmark coordinates and a gesture label, but no participant identifier, recording-session identifier, source-video identifier, or capture-device identifier. Therefore, the repository cannot establish whether related frames or repeated captures from the same person/session appear in both partitions.

Do not describe the current result as participant-independent, session-independent, or real-world accuracy. Do not claim data leakage has been ruled out.

## Required data collection metadata for a stronger test

For every newly collected sample, maintain a separate metadata table with:

- `sample_id`: stable unique row identifier.
- `participant_group`: random pseudonymous code, not a name or other direct identifier.
- `session_group`: random code for one recording session.
- `capture_device_group`: optional pseudonymous device category/code.
- `gesture_label`: one of the five supported labels.
- `capture_conditions`: brief non-identifying tags such as lighting or camera angle.

Do not store faces, names, contact details, or other direct identifiers in this metadata. Obtain consent before collecting another person's hand/camera data. Keep metadata access limited and document retention.

## Recommended evaluation design

1. Collect multiple independent participants and multiple sessions per participant.
2. Keep all rows from a participant in exactly one partition for the primary participant-independent evaluation. Use a group-aware split such as `GroupShuffleSplit` or `StratifiedGroupKFold`, with participant code as the group.
3. If the intended use includes new recording sessions from known participants, report a separate session-held-out evaluation using session code as the group.
4. Fit preprocessing, feature selection, hyperparameter tuning, and model selection using training data only. Do not use the final test partition to choose a model.
5. Keep the final test set untouched until the model and thresholds are fixed.
6. Report accuracy, macro F1, per-class precision/recall, confusion matrix, group counts, class counts, and confidence intervals where appropriate.
7. Compare performance across lighting, camera angle, device, and other documented capture conditions. Report small subgroup sizes.
8. Run a real webcam smoke test on supported hardware and record the environment separately from offline model metrics.

## Current project status

The present CSV files do not contain the group identifiers needed for steps 2 and 3. Do not synthesize participant or session IDs from row order. Until a properly grouped dataset is collected, retain the current 98% figure only as a reproducible random-row holdout benchmark.

## Reproduction

The current row-level evaluation is reproducible with:

```bash
python src/evaluate_committed_model.py
python src/save_model.py
```

Both scripts use the same deterministic stratified random split. The second command overwrites the local model artifact, so preserve the existing model and reports before running it if you need to compare versions.
