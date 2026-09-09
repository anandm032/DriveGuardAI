# Datasets

This folder holds the data used to train the YOLO driver-behavior model.

- `raw/` — datasets as downloaded/collected, unmodified.
- `processed/` — cleaned, annotated, and split into train/val/test in
  YOLO format, ready for `training/train.py`.

Nothing is populated yet — dataset selection and annotation happens in
**Phase 5/6 (Dataset and Training)**. When we get there, this README
will be updated with:

1. Exact dataset name(s) used
2. Where each was obtained (source/link)
3. Which classes each contains
4. Annotation format and tool used (e.g. Roboflow, CVAT)
5. License terms for each dataset

No dataset sizes, class counts, or accuracy numbers should be assumed
here until we've actually selected and inspected the data.
