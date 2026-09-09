# Datasets

This folder holds the data used to train the YOLO driver-behavior model.

- `raw/` — datasets as downloaded, unmodified.
- `processed/` — cleaned, annotated, and split into train/val/test in
  YOLO format (images/ + labels/ per split), ready for `training/train.py`.

Nothing is downloaded yet — that's a manual step for you, since it
needs a Roboflow account and a browser. This README documents real,
currently-available options found while researching this phase, with
honest image counts as reported by each source. **None of these
numbers are guaranteed accurate forever** — dataset hosts update their
collections over time, so re-check the linked page before you commit
to one.

## Practical class list for this project

To keep Phase 6 achievable, DriveGuard AI starts with 5 classes,
deliberately matching the activity names already used in
`config.yaml`'s `penalties` section (so no translation layer is
needed between the model's output and the scoring engine):

```
mobile_phone_usage
smoking
drinking
distraction
drowsiness
```

`drowsiness` is detected and displayed but intentionally excluded
from scoring by default (see `scoring/safety_score.py` — this mirrors
spec section 10). Eating was in the original wishlist but is dropped
from the *first* training run to keep class count practical, per the
spec's own instruction to start smaller if training everything at
once is too hard — it can be added back in later by re-running
`training/train.py` once you have annotated eating examples,
without changing any other module.

## Recommended datasets (found via web search, current as of this phase)

### Option A — single combined "Driver Monitoring System" dataset (recommended starting point)

Several independent Roboflow Universe projects (and a companion
open-source GitHub project) use almost exactly this project's target
classes already, which saves you from merging multiple datasets by
hand:

- **GitHub — AlbatrossC/Driver-Monitoring-System**
  (`https://github.com/AlbatrossC/Driver-Monitoring-System`) — an
  open-source driver monitoring project whose own class list is
  `Distracted, Drinking, Drowsy, Eating, PhoneUse, SafeDriving,
  Seatbelt, Smoking` — an almost exact match to this project's needs.
  It documents which underlying Roboflow datasets it combined and how
  it preprocessed them (cleaning, augmentation, class balancing, YOLO
  format conversion) — worth reading even if you don't reuse its
  exact pipeline.
- **Roboflow Universe — "Driver Monitoring system" (SnY)**
  (`https://universe.roboflow.com/sny-bdtx6/driver-monitoring-system-v0mei-e3fa5`)
  — reported ~4,000 images, 7 classes: Distracted, Drinking, Drowsy,
  Eating, Mobile use, SafeDriving, Smoking.
- **Roboflow Universe — "Driver Monitoring system" (Shayari
  Bhattacharjee)** — a near-duplicate of the above with the same class
  set, reported ~4,023 images.
- **Roboflow Universe — "Driver monitoring system" (JNU)**
  (`https://universe.roboflow.com/jnu-k4vdw/driver-monitoring-system-nwzb4`)
  — reported ~8,183 images.

**Why this is the recommended starting point:** one dataset covering
nearly every target class means less manual merging and fewer class-
naming mismatches. Browse a couple of these on Roboflow Universe
first, preview the actual images (quality/lighting/camera angle
varies a lot between them), and pick the one whose driving position
and camera angle looks closest to a normal laptop webcam pointed at a
driver — that's a judgment call only you can make by looking at the
previews yourself, not something to take on faith from an image count.

### Option B — per-activity datasets (use if Option A's coverage or quality is insufficient)

- **Mobile phone usage / distraction:**
  - Roboflow — `ipylot-project/distracted-driving-v2wk5` — reported
    ~7,000 training / 1,000 val / 1,000 test images, 12 classes
    (documented in a Medium walkthrough by Sam Ansari on how to train
    a YOLOv5 model on it).
  - Roboflow — `yolov8-z7kip/distracted-driver-detection-bvtnl` —
    reported ~1,216 images, 2 classes (Attentive / Distracted) —
    coarser labeling, useful as a supplement not a primary source.
  - The classic **State Farm Distracted Driver Detection** dataset
    (Kaggle competition) is frequently cited alongside these as a
    benchmark — it's image classification (whole-image labels), not
    object-detection bounding boxes, so it needs manual bounding-box
    annotation before it's usable for YOLO. Several papers (e.g. the
    ME-YOLOv8 driver-distraction paper on IET Intelligent Transport
    Systems) use it only for benchmarking, not for the YOLO training
    set itself, for that exact reason.

- **Drowsiness:**
  - Roboflow — `testdemo-orvjj/driver-drowsiness-detection-qe2v0` —
    reported ~969 images, yawning/sleeping classes.
  - The **YawDD** video dataset and **UTA-RLDD** dataset are commonly
    used in published drowsiness-detection work (referenced in a
    Medium writeup on YOLOv5 drowsiness detection) — these are video
    datasets, so frames need to be extracted and annotated before use.
  - Typical class sets across these drowsiness datasets: `open_eye`,
    `closed_eye`, `yawn`, sometimes `head_nod` — during annotation,
    map all of these to a single `drowsiness` class for this project
    (start simple; per-symptom detail like "eyes closed vs yawning"
    can be a future enhancement, not a Phase 6 requirement).

- **Smoking:**
  - Roboflow — `cigarettedetection/cigarette_detection` — reported
    ~873 images, single `cigarette` class.
  - Roboflow — `nehal-lsski/cigarette-detection-y1xgi-szmbd` —
    reported ~8,666 images.
  - Roboflow — `cigarette-7k8kb/cigarette-2uvvq` — reported ~1,451
    images, includes a pre-trained model you can compare against.
  - Most cigarette datasets label the **object** ("cigarette"), not
    the **action** — during annotation, treat "a driver with a
    cigarette bounding box present" as the `smoking` class label for
    this project, consistent with how the combined Option A datasets
    already do it.

## Annotation workflow (if you need to relabel or combine datasets)

1. Create a free Roboflow account at roboflow.com and start a new
   Object Detection project.
2. Upload the raw images (or import an existing Roboflow Universe
   dataset directly into your own workspace with its "Fork Dataset"
   button — this avoids re-annotating from scratch).
3. **Rename classes to this project's canonical 5 names** listed
   above (`mobile_phone_usage`, `smoking`, `drinking`, `distraction`,
   `drowsiness`) using Roboflow's class remapping/renaming tool during
   the "Generate" step — this is what keeps `detector.py`'s output
   directly usable by `scoring/safety_score.py` with no translation
   step. If you'd rather keep the dataset's original class names
   (e.g. "Mobile use", "Distracted", "Drowsy") instead of renaming
   them, that also works without extra effort — `config.yaml`'s
   `model.class_map` already translates those common raw names to the
   canonical ones at inference time, so skipping this renaming step
   is a valid shortcut, not a requirement.
4. Apply Roboflow's train/valid/test split (75/15/10 or similar —
   whatever the source dataset already used is a reasonable default
   unless you have a specific reason to change it).
5. Export in **"YOLOv8"** (or "YOLO v8 PyTorch") format, download the
   zip, and unzip it into `datasets/processed/` so the folder looks
   like:
   ```
   datasets/processed/
   ├── train/images/  train/labels/
   ├── valid/images/  valid/labels/
   ├── test/images/   test/labels/
   └── data.yaml      (Roboflow generates one - compare it against
                        training/data.yaml below and reconcile class
                        names/order if they differ)
   ```
   If you prefer manual annotation instead of using an existing
   dataset, CVAT (cvat.ai) is a solid free alternative to Roboflow —
   export from CVAT in "YOLO 1.1" format and place the result in the
   same folder structure.

## What we are NOT doing in Phase 6

- Not fabricating any accuracy/mAP/FPS numbers — `training/evaluate.py`
  (this phase) computes these for real once you've actually trained on
  real data. Until then, any number would be invented.
- Not requiring you to train on all 5 classes at once if that proves
  too slow/hard on your hardware — `training/data.yaml` is a plain
  list, so dropping a class to iterate faster is a one-line edit.
- Not committing dataset images to git — see `.gitignore`; only this
  README and `data.yaml` (structure, not content) belong in version
  control.
