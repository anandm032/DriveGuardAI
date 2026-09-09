# DriveGuard AI

**Intelligent Driver Behavior Monitoring and Insurance Risk Scoring System**

Final-year B.Tech project (Artificial Intelligence & Data Science).

## What this project does

DriveGuard AI uses a laptop webcam + a YOLO-based deep learning model to
monitor a driver in real time, detect unsafe driving activities (mobile
phone usage, smoking, drinking, distraction, drowsiness), and:

1. Confirms violations using temporal filtering (to avoid false positives
   from a single noisy frame).
2. Deducts points from a persistent **Vehicle Safety Score** (starts at
   100, stored in SQLite, never resets between sessions).
3. Classifies the vehicle into **Low / Medium / High** risk based on
   configurable, project-defined thresholds (clearly **not** an official
   insurance industry standard).
4. Displays everything on a Streamlit dashboard: live webcam feed,
   current activity + confidence, safety score, risk level, violation
   history, and charts.
5. Generates an **Insurance Risk Evaluation** — a recommendation, not a
   real insurance decision or premium change.

The system is designed to run entirely on a normal laptop. An MQ-2 smoke
sensor is supported as an **optional** future hardware add-on for smoking
verification — it is never required to run the core prototype.

## Project status

This repository is being built incrementally, phase by phase. See
`PHASES.md` (added once later phases begin) for progress. Phase 1
(project setup) is complete.

## Folder structure

```
DriveGuardAI/
├── app.py                  # main entry point / launcher
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
│
├── config/
│   └── config.yaml         # all tunable settings (model, thresholds, penalties, DB path)
│
├── models/
│   └── best.pt             # trained YOLO weights (added in Phase 6, not committed to git)
│
├── datasets/
│   ├── raw/                # raw downloaded/collected datasets
│   ├── processed/          # cleaned + annotated data ready for training
│   └── README.md           # dataset sources and licensing notes
│
├── training/
│   ├── train.py            # YOLO training script (Phase 6)
│   ├── evaluate.py         # precision/recall/F1/mAP evaluation (Phase 6)
│   └── data.yaml           # YOLO dataset definition
│
├── detection/
│   ├── detector.py         # loads YOLO model, runs inference on a frame (Phase 6)
│   ├── webcam.py           # webcam capture loop (Phase 7)
│   └── violation_confirmation.py  # temporal confirmation logic (Phase 8)
│
├── scoring/
│   ├── safety_score.py     # safety score engine (Phase 4)
│   └── risk_classifier.py  # risk level classification (Phase 5)
│
├── database/
│   ├── database.py         # SQLite connection + queries (Phase 3)
│   ├── models.py           # table definitions
│   └── schema.sql          # raw SQL schema
│
├── dashboard/
│   └── dashboard.py        # Streamlit dashboard (Phase 11)
│
├── reports/
│   └── report_generator.py # PDF/CSV report generation (Phase 12)
│
├── utils/
│   ├── logger.py           # logging setup
│   └── helpers.py          # config loading and shared helpers
│
├── tests/
│   ├── test_scoring.py
│   ├── test_database.py
│   └── test_risk.py
│
├── scripts/
│   └── health_check.py     # verifies the environment is set up correctly
│
└── data/
    └── driveguard.db       # SQLite database file (created at first run)
```

## Requirements

- Python 3.10 or 3.11 (recommended)
- A working webcam for the real-time detection phases
- Windows 10/11 (setup commands below target Windows; the project is
  pure Python so it also runs on macOS/Linux with the equivalent venv
  commands)

## Setup (Windows)

```powershell
# 1. Navigate into the project folder
cd DriveGuardAI

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

## Verify the environment

```powershell
python scripts\health_check.py
```

This checks that OpenCV, PyTorch, TorchVision, Ultralytics, Streamlit,
NumPy, Pandas, and PyYAML are all importable, and tries to open your
webcam. It does **not** load a YOLO model yet — that's added in Phase 6.

## Run the app skeleton

```powershell
python app.py
```

This currently just confirms the configuration file loads correctly.
Real functionality (database, detection, dashboard) is added phase by
phase.

## Privacy note

This system processes webcam frames locally. No video is uploaded to
external services. Only violation metadata (activity type, confidence,
timestamp, penalty) is stored in the local SQLite database — not raw
video — unless a future phase explicitly adds video storage with your
consent.

## Important disclaimers

- Risk thresholds (Low/Medium/High) are **project-defined policy** for
  demonstration purposes, not an official insurance industry standard.
- The "Insurance Risk Evaluation" is a **recommendation**, not a real
  insurance decision, premium change, or integration with any actual
  insurer.
- All dataset sizes, model accuracy, mAP, and FPS figures reported by
  this project come from actual measurement scripts — none are
  fabricated.
