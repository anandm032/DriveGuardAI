import cv2
from detection.detector import Detector
from utils.helpers import load_config

config = load_config()
detector = Detector(config=config)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    raise SystemExit(1)

print("Webcam test started.")
print("Show your phone to the camera.")
print("Press Q in the camera window to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read webcam frame.")
        break

    detections = detector.detect(frame)

    for detection in detections:
        print(
            f"Detected: {detection.activity} "
            f"| confidence={detection.confidence:.2f}"
        )

        bbox = detection.bbox
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"{detection.activity} {detection.confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("DriveGuard AI - Detector Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test stopped.")
