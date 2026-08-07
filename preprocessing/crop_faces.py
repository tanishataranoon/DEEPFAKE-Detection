"""
=========================================================
Face Cropper
---------------------------------------------------------
Crop tracked faces and resize every crop to 224×224.

Both the Appearance Branch and Temporal Branch will use
these resized face crops.
=========================================================
"""

import cv2

from collections import defaultdict

from preprocessing.data_structures import FaceCrop


class FaceCropper:

    def __init__(self):

        # track_id -> list[FaceCrop]
        self.face_history = defaultdict(list)

    def update(self, frame, tracks):

        crops = []

        for track in tracks:

            x1, y1, x2, y2 = map(int, track.bbox)

            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(
                face,
                (224, 224),
                interpolation=cv2.INTER_LINEAR
            )

            face_height, face_width = face.shape[:2]

            crop = FaceCrop(
                track_id=track.track_id,
                frame_id=track.frame_id,
                bbox=track.bbox,
                confidence=track.confidence,
                face_width=face_width,
                face_height=face_height
            )

            crops.append(crop)

            self.face_history[track.track_id].append(crop)

        return crops

    def get_face_history(self):

        return self.face_history