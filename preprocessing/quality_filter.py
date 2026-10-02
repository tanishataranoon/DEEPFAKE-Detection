"""
=========================================================
Quality Filter
---------------------------------------------------------
Removes poor-quality detections before tracking.
=========================================================
"""

import cv2

from config import CONFIDENCE_THRESHOLD


class QualityFilter:

    def __init__(
        self,
        min_face_size=19,
        blur_threshold=16
    ):
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.reset_stats()


    def reset_stats(self):
        self.total = 0
        self.accepted = 0

        self.reject_confidence = 0
        self.reject_width = 0
        self.reject_height = 0
        self.reject_blur = 0
        self.reject_empty = 0


    def get_stats(self):

        rejected = (
            self.reject_confidence
            + self.reject_width
            + self.reject_height
            + self.reject_blur
            + self.reject_empty
        )

        total = self.accepted + rejected

        return {
                "total": total,              # was self.total
                "accepted": self.accepted,
                "reject_width": self.reject_width,
                "reject_height": self.reject_height,
                "reject_blur": self.reject_blur,
                "reject_confidence": self.reject_confidence,
                "reject_empty": self.reject_empty,   # was missing from the returned dict entirely
            }
    def _blur_score(self, face):

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    def filter(self, frame, detections):

        filtered = []

        for det in detections:

            x1, y1, x2, y2 = map(int, det.bbox)

            # Confidence check
            if det.confidence < CONFIDENCE_THRESHOLD:
                self.reject_confidence += 1
                continue

            # Face size check
            if (x2 - x1) < self.min_face_size:
                self.reject_width += 1
                continue

            if (y2 - y1) < self.min_face_size:
                self.reject_height += 1
                continue

            # Crop face
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                self.reject_empty += 1
                continue

            # Blur check
            blur = self._blur_score(face)

            if blur < self.blur_threshold:
                self.reject_blur += 1
                continue

            # Passed every test
            self.accepted += 1
            filtered.append(det)

        return filtered
