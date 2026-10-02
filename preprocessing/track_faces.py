from collections import defaultdict
import numpy as np
from supervision import ByteTrack, Detections

from preprocessing.data_structures import Track


class FaceTracker:

    def __init__(self):

        self.tracker = ByteTrack()

        # Stores all tracked observations by ID
        self.track_history = defaultdict(list)

    def update(self, frame_id, detections):

        if not detections:
            return []

        xyxy = np.array(
            [det.bbox for det in detections],
            dtype=np.float32
        )

        confidence = np.array(
            [det.confidence for det in detections],
            dtype=np.float32
        )

        class_id = np.zeros(len(detections), dtype=np.int32)

        sv_detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        tracked = self.tracker.update_with_detections(
            sv_detections
        )

        tracks = []

        if tracked.tracker_id is None:
            return tracks

        for i in range(len(tracked.tracker_id)):

            track = Track(
                track_id=int(tracked.tracker_id[i]),
                frame_id=frame_id,
                bbox=tracked.xyxy[i].tolist(),
                confidence=float(tracked.confidence[i])
            )

            tracks.append(track)

            # Save this observation for later
            self.track_history[track.track_id].append(track)

        return tracks

    def get_track_history(self):
        """
        Returns:
            {
                1: [Track, Track, ...],
                2: [Track, Track, ...]
            }
        """
        return self.track_history