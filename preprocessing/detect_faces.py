"""
=========================================================
YOLOv11 Face Detector
=========================================================
"""

from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE
)
from preprocessing.data_structures import Detection

class FaceDetector:

    def __init__(self):

        self.model = YOLO(str(YOLO_MODEL))

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                detections.append(
                   Detection(
                       bbox=[x1,y1,x2,y2],
                       confidence=confidence
                   )
               )

        return detections

