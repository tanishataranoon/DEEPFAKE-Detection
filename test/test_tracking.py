import cv2

from preprocessing.extract_frame import extractframe
from preprocessing.detect_faces import FaceDetector
from preprocessing.quality_filter import QualityFilter
from preprocessing.track_faces import FaceTracker
from preprocessing.crop_faces import FaceCropper
from preprocessing.creat_clip import ClipCreator
from preprocessing.build_metadata import MetadataBuilder

# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

reader = extractframe(
    "data/raw/DFD_original sequences/01__exit_phone_room.mp4"
)

detector = FaceDetector()
quality = QualityFilter()
tracker = FaceTracker()
cropper = FaceCropper()


# ---------------------------------------------------
# Process Video
# ---------------------------------------------------

for frame_data in reader:

    frame = frame_data.image

    detections = detector.detect(frame)
    detections = quality.filter(frame, detections)

    tracks = tracker.update(
        frame_data.frame_id,
        detections
    )

    crops = cropper.update(
        frame,
        tracks
    )

    # Visualization
    for t in tracks:

        x1, y1, x2, y2 = map(int, t.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"ID {t.track_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break


# ---------------------------------------------------
# Build Clips
# ---------------------------------------------------

face_history = cropper.get_face_history()

clip_creator = ClipCreator(
    clip_length=16
)

clips = clip_creator.create(face_history)


# ---------------------------------------------------
# Print Summary
# ---------------------------------------------------

print("\n" + "="*60)
print("TRACK SUMMARY")
print("="*60)

track_history = tracker.get_track_history()

for track_id, track_list in track_history.items():

    print(
        f"Track {track_id} : "
        f"{track_list[0].frame_id} -> "
        f"{track_list[-1].frame_id} "
        f"({len(track_list)} frames)"
    )


print("\n" + "="*60)
print("FACE HISTORY")
print("="*60)

for track_id, faces in face_history.items():

    print(
        f"Track {track_id} : "
        f"{len(faces)} cropped faces"
    )


print("\n" + "="*60)
print("FACE CLIPS")
print("="*60)

for track_id, clip_list in clips.items():

    print(
        f"Track {track_id} : "
        f"{len(clip_list)} clips"
    )

    for clip in clip_list:

        print(
            f"   Clip {clip.clip_id} "
            f"({len(clip.frames)} frames)"
        )

metadata = MetadataBuilder()

metadata.add_video(
    clips=clips,
    video_path="data/raw/DFD_original sequences/01__exit_phone_room.mp4",
    label="real"
)

metadata.save(
    "data/processed/metadata.csv"
)
from preprocessing.split_dataset import DatasetSplitter

splitter = DatasetSplitter()

splitter.split(
    metadata_csv="data/processed/metadata.csv",
    output_dir="data/split"
)
reader.release()
cv2.destroyAllWindows()