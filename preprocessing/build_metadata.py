"""
=========================================================
Build Metadata
---------------------------------------------------------
Converts FaceClip objects into metadata.csv.

This metadata is later used by the training dataloader to
reconstruct each temporal clip directly from the original
video.
=========================================================
"""

import os
import pandas as pd


class MetadataBuilder:

    def __init__(self):

        self.rows = []

    # =====================================================
    # Add FaceClip objects
    # =====================================================

    def add_clips(self, clips):
        """
        clips : list[FaceClip]
        """

        for clip in clips:

            self.rows.append(
                {
                    # -----------------------------------
                    # Video Information
                    # -----------------------------------
                    "video_name": clip.video_name,
                    "label": clip.label,
                    # -----------------------------------
                    # Track Information
                    # -----------------------------------
                    "track_id": clip.track_id,
                    "clip_id": clip.clip_id,
                    # -----------------------------------
                    # Temporal Information
                    # -----------------------------------
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "fps": clip.fps,
                    "num_frames": clip.num_frames,
                    "clip_duration": clip.clip_duration,
                    "video_duration": clip.video_duration,
                    # -----------------------------------
                    # Appearance Information
                    # -----------------------------------
                    "face_width": clip.face_width,
                    "face_height": clip.face_height,
                    "confidences": ",".join(f"{c:.4f}" for c in clip.confidences),
                    # -----------------------------------
                    # Reconstruction Information
                    # -----------------------------------
                    "frame_ids": ",".join(map(str, clip.frame_ids)),
                    "bboxes": str(clip.bboxes),
                }
            )

    # =====================================================
    # Save Metadata
    # =====================================================

    def save(self, output_csv):

        output_dir = os.path.dirname(output_csv)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.rows)

        df.to_csv(output_csv, index=False)

        print(f"Metadata updated : {len(df)} clips")

        print(f"Saved to         : {output_csv}")
