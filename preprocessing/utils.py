"""
=========================================================
Utility Functions
---------------------------------------------------------
Common helper functions used by preprocessing modules.
=========================================================
"""

from pathlib import Path
import logging

# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Directory Utilities
# ---------------------------------------------------------

def create_directory(path: Path):
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Video Utilities
# ---------------------------------------------------------

def get_video_files(root_directory: Path):
    """
    Recursively find all videos inside a directory.

    Returns:
        list[Path]
    """

    extensions = (".mp4", ".avi", ".mov", ".mkv")

    videos = []

    for ext in extensions:
        videos.extend(root_directory.rglob(f"*{ext}"))

    videos.sort()

    logger.info(f"Found {len(videos)} videos.")

    return videos
