# Diff-EEG

A diffusion-based EEG foundation model for seizure detection and seizure subtype classification.

## Real-Time EEG Seizure Detection Demo

<video src="https://github.com/abdulkader902017/Diff-EEG/raw/main/output.mp4" controls autoplay muted width="100%"></video>

## Repository Structure

```
├── Diff_EEG_train.py                  # Pre-training script (diffusion backbone)
├── training_history.json              # Pre-training metrics
├── training_progress.png              # Pre-training curves
├── output.mp4                         # Real-time inference demo
└── evaluation/
    ├── binary_seizure_detection/
    │   └── patientwise_binary_classification.py
    └── subtype_classification/
        ├── segment_wise_stratified/   # Stratified 80/20 split
        ├── segment_wise_fewshot/      # Few-shot (frozen vs unfrozen)
        └── patient_wise_cv/           # 5-fold patient-wise CV
```