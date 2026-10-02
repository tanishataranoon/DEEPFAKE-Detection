"""
=========================================================
Test Pipeline

VideoReader
↓

FaceDetector
↓

QualityFilter

Press ESC to exit.
=========================================================
"""

import cv2

from preprocessing.extract_frame import extractframe
from preprocessing.detect_faces import FaceDetector
from preprocessing.quality_filter import QualityFilter


reader = extractframe(
    "data/raw/DFD_original sequences/01__exit_phone_room.mp4"
)

detector = FaceDetector()

quality = QualityFilter()
total_frames = 0
frames_with_face = 0
total_faces = 0

for frame_data in reader:

    frame = frame_data.image

    detections = detector.detect(frame)

    detections = quality.filter(
        frame,
        detections
    )
    print(detector.model.names)
    print(
        f"Frame {frame_data.frame_id} -> {len(detections)} faces"
    )

    # Draw detections

    #for det in detections:
        #x1, y1, x2, y2, conf = det
    for det in detections:

        x1, y1, x2, y2 = map(int, det.bbox)
        conf = det.confidence

        cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0,255,0),
        2
        )

        cv2.putText(
        frame,
        f"{conf:.2f}",
        (int(x1), int(y1)-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
        )

        cv2.imshow(
        "Detection Pipeline",
        frame
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

        total_frames += 1

        if len(detections) > 0:
            frames_with_face += 1

        total_faces += len(detections)
reader.release()
print("\n========== Detection Summary ==========")
print(f"Total Frames        : {total_frames}")
print(f"Frames with Faces   : {frames_with_face}")
print(f"Frames without Face : {total_frames - frames_with_face}")
print(f"Total Detections    : {total_faces}")

print(
    f"Detection Rate      : "
    f"{100 * frames_with_face / total_frames:.2f}%"
)
cv2.destroyAllWindows()
