"""
=========================================================
Full Preprocessing Pipeline
---------------------------------------------------------
Video
    ↓
Frame Extraction
    ↓
YOLO Face Detection
    ↓
Quality Filter
    ↓
ByteTrack Tracking
    ↓
Face Cropping (224×224)
    ↓
Clip Creation (Metadata Only)
    ↓
Metadata Builder
=========================================================
"""

import os
import cv2
import time
import pandas as pd

from preprocessing.build_metadata import MetadataBuilder
from preprocessing.split_dataset import DatasetSplitter
from preprocessing.extract_frame import extractframe
from preprocessing.detect_faces import FaceDetector
from preprocessing.quality_filter import QualityFilter
from preprocessing.track_faces import FaceTracker
from preprocessing.crop_faces import FaceCropper
from preprocessing.creat_clip import ClipCreator


class PreprocessPipeline:

    def __init__(self, clip_length=16):
        self.detector = FaceDetector()
        self.quality = QualityFilter()
        self.clip_creator = ClipCreator(clip_length=clip_length)

    # =====================================================
    # Process One Video
    # =====================================================

    def process_video(self, video_path, label, metadata):

        print(f"\nProcessing : {os.path.basename(video_path)}")

        tracker = FaceTracker()
        cropper = FaceCropper()
        self.quality.reset_stats()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = video_frame_count / fps if fps > 0 else 0
        cap.release()

        reader = extractframe(video_path)

        total_frames = 0
        for frame_data in reader:

            total_frames += 1
            frame = frame_data.image

            detections = self.detector.detect(frame)
            detections = self.quality.filter(frame, detections)
            tracks = tracker.update(frame_data.frame_id, detections)
            cropper.update(frame, tracks)

        reader.release()

        track_history = tracker.get_track_history()
        face_history = cropper.get_face_history()

        clips = self.clip_creator.create(
            face_history=face_history,
            video_name=os.path.basename(video_path),
            label=label,
            fps=fps,
            video_duration=video_duration,
        )

        stats = self.quality.get_stats()
        metadata.add_clips(clips)

        acceptance_rate = (
            100 * stats["accepted"] / stats["total"] if stats["total"] > 0 else 0
        )

        print()
        print(f"Frames          : {total_frames}")
        print(f"Detections      : {stats['total']}")
        print(f"Accepted        : {stats['accepted']}")
        print(f"Rejected Width  : {stats['reject_width']}")
        print(f"Rejected Height : {stats['reject_height']}")
        print(f"Rejected Blur   : {stats['reject_blur']}")
        print(f"Rejected Conf   : {stats['reject_confidence']}")
        print(f"Acceptance Rate : {acceptance_rate:.1f}%")
        print(f"Tracks          : {len(track_history)}")
        print(f"Clips           : {len(clips)}")

    # =====================================================
    # Process Entire Dataset
    # =====================================================

    def process_dataset(self, dataset_root):

        metadata = MetadataBuilder()
        metadata_path = "data/processed/metadata.csv"

        # -----------------------------------------
        # Load existing metadata instead of deleting it
        # -----------------------------------------
        existing_video_names = set()

        if os.path.exists(metadata_path):
            existing_df = pd.read_csv(metadata_path)
            metadata.rows = existing_df.to_dict("records")
            existing_video_names = set(existing_df["video_name"].unique())

            print(f"Loaded existing metadata: {len(existing_df)} clips "
                  f"from {len(existing_video_names)} videos")

        # dataset now defined unconditionally, not nested in the if-block
        dataset = {
            "real": "DFD_original sequences",
            "fake": os.path.join("DFD_manipulated_sequences", "DFD_manipulated_sequences"),
        }

        total_videos = 0
        for folder_name in dataset.values():
            folder = os.path.join(dataset_root, folder_name)
            total_videos += len(
                [v for v in os.listdir(folder) if v.lower().endswith(".mp4")]
            )

        video_counter = 0
        start_time = time.time()

        for label, folder_name in dataset.items():

            folder = os.path.join(dataset_root, folder_name)

            if not os.path.exists(folder):
                raise FileNotFoundError(folder)

            print("\n" + "=" * 60)
            print(f"PROCESSING {label.upper()} VIDEOS")
            print("=" * 60)

            videos = sorted(os.listdir(folder))

            for video in videos:

                if not video.lower().endswith(".mp4"):
                    continue

                video_counter += 1

                if video in existing_video_names:
                    print(f"\nSkipping {video} (already processed) "
                          f"[{video_counter}/{total_videos}]")
                    continue

                print("\n" + "=" * 60)
                print(f"Video {video_counter} / {total_videos}")
                print("=" * 60)

                self.process_video(
                    video_path=os.path.join(folder, video),
                    label=label,
                    metadata=metadata,
                )

                metadata.save(metadata_path)

                elapsed = time.time() - start_time
                avg_time = elapsed / video_counter
                remaining = avg_time * (total_videos - video_counter)

                print()
                print(f"Elapsed         : {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                print(f"Remaining       : {time.strftime('%H:%M:%S', time.gmtime(remaining))}")

    

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)