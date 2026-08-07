"""
=========================================================
Create Fixed-Length Face Clips
---------------------------------------------------------
Creates temporal clips from tracked faces.

This module DOES NOT save image clips.

It only creates FaceClip metadata that will later be
written into metadata.csv.

During training, the dataloader will reconstruct every
clip directly from the original video.
=========================================================
"""

from preprocessing.data_structures import FaceClip


class ClipCreator:

    def __init__(self, clip_length=16):

        self.clip_length = clip_length

    def create(
        self,
        face_history,
        video_name,
        label,
        fps,
        video_duration
    ):
        """
        Parameters
        ----------
        face_history : dict
            {
                track_id : [FaceCrop, FaceCrop, ...]
            }

        video_name : str

        label : str
            "real" or "fake"

        Returns
        -------
        list[FaceClip]
        """

        clips = []

        # ---------------------------------------------
        # Create clips for every tracked face
        # ---------------------------------------------

        for track_id, faces in face_history.items():

            faces = sorted(
                faces,
                key=lambda x: x.frame_id
            )

            clip_id = 0

            for start in range(
                0,
                len(faces),
                self.clip_length
            ):

                end = start + self.clip_length

                # Ignore incomplete clips
                if end > len(faces):
                    break
                num_frames = end - start

                clip_duration = (
                    num_frames / fps
                    if fps > 0
                    else 0
                )
                clip_faces = faces[start:end]

                clip = FaceClip(
                    track_id=track_id,
                    clip_id=clip_id,
                    video_name=video_name,
                    fps=fps,
                    video_duration=video_duration,
                    label=label,
                    start_frame=faces[start].frame_id,
                    end_frame=faces[end - 1].frame_id,
                    frame_ids=[f.frame_id for f in faces[start:end]],
                    bboxes=[f.bbox for f in faces[start:end]],
                    num_frames=len(faces[start:end]),
                    confidences=[f.confidence for f in faces[start:end]],
                    face_width=int(
                        sum(f.face_width for f in faces[start:end])
                        / len(faces[start:end])
                    ),
                    clip_duration=clip_duration,
                    face_height=int(
                        sum(f.face_height for f in faces[start:end])
                        / len(faces[start:end])
                    ),
                )

                clips.append(clip)

                clip_id += 1

        return clips
