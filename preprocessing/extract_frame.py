"""
=========================================================
Video Reader
---------------------------------------------------------
Reads a video frame-by-frame.

Frames are NOT saved.

Each frame is streamed to the next module.
=========================================================
"""

import cv2
from preprocessing.data_structures import FrameData

class extractframe:

    def __init__(self, video_path):

        self.video_path = str(video_path)

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():

            raise FileNotFoundError(
                f"Cannot open video: {video_path}"
            )

        self.frame_index = 0

    def __iter__(self):

        return self

    def __next__(self):

        success, frame = self.cap.read()

        if not success:

            self.cap.release()

            raise StopIteration

        self.frame_index += 1

        return FrameData(
            frame_id=self.frame_index,
            image=frame
        )

    def release(self):

        self.cap.release()
