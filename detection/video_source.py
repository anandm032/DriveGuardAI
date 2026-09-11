"""
DriveGuard AI - Video Source (Webcam / Video File)
======================================================
A thin, uniform wrapper around cv2.VideoCapture that treats a webcam
and a video file the same way from the caller's perspective: open it,
iterate frames, close it. Phase 7.3's MonitoringSession is the
intended consumer - this module exists so that orchestration code
never has to know or care whether it's reading from a live camera or
a test video file.

This module is deliberately independent of everything else in the
project - no Detector, no ViolationTracker, no database, no scoring.
It only knows about OpenCV and frames. That's what makes it testable
without a webcam: point it at a small video file instead.

Usage:

    from detection.video_source import VideoSource

    # Webcam (index 0 is almost always the default/built-in camera)
    with VideoSource(0) as source:
        print(source.metadata)
        for frame in source:
            ...  # frame is a BGR numpy array, same as cv2.VideoCapture.read()

    # Video file, for testing without a camera
    with VideoSource("sample_drive.mp4") as source:
        for frame in source:
            ...

Iteration stops (StopIteration) at end-of-file for a video file, or if
a read ever fails (e.g. a disconnected webcam) - callers that need to
tell "clean EOF" apart from "camera dropped mid-stream" should check
`source.source_type` (a file ending is expected; a webcam ending
usually isn't) - this module doesn't guess which one happened, it
just reports "no more frames" either way, consistent with how
cv2.VideoCapture.read() itself reports both cases identically.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union

import cv2

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoSourceError(Exception):
    """Raised when a video source can't be opened, or is used before
    being opened / after being closed. Never raised for a normal
    end-of-file - that's reported by read()/iteration simply
    producing no more frames, not an error."""
    pass


@dataclass
class SourceMetadata:
    source_type: str            # "webcam" or "file"
    source: str                 # str(camera index) or file path
    width: int
    height: int
    fps: float
    frame_count: Optional[int]  # total frames if known (files); None for webcams (unbounded/live)


class VideoSource:
    def __init__(self, source: Union[int, str]):
        """source: an int camera index (e.g. 0 for the default
        webcam) or a str path to a video file. Nothing is opened yet -
        call open() explicitly, or use this object as a context
        manager (`with VideoSource(...) as vs:`), which is the
        recommended way since it guarantees release() runs even if
        the caller's loop raises partway through."""
        self.source = source
        self.source_type = "webcam" if isinstance(source, int) else "file"
        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[SourceMetadata] = None

    def open(self) -> "VideoSource":
        """Open the underlying camera/file. Raises VideoSourceError
        with a clear, specific message on failure - never silently
        proceeds with a half-open capture.

        Idempotent: if already open, this is a no-op that returns
        self - it never constructs a second VideoCapture over an
        existing one (which would leak the first handle and silently
        lose track of it)."""
        if self.is_opened():
            logger.debug(
                f"VideoSource.open() called while already open "
                f"(type={self.source_type}, source={self.source}) - no-op."
            )
            return self

        if self.source_type == "file" and not os.path.exists(self.source):
            raise VideoSourceError(f"Video file not found: {self.source}")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            if self.source_type == "webcam":
                raise VideoSourceError(
                    f"Failed to open webcam at index {self.source}. "
                    f"Check that a camera is connected, not already in use by "
                    f"another application, and that this app has camera "
                    f"permission."
                )
            else:
                raise VideoSourceError(
                    f"Failed to open video file: {self.source}. "
                    f"The file may be corrupt, empty, or in a codec OpenCV "
                    f"can't decode on this machine."
                )

        self._cap = cap
        self.metadata = self._read_metadata()
        logger.info(
            f"VideoSource opened: type={self.source_type}, source={self.source}, "
            f"{self.metadata.width}x{self.metadata.height} @ {self.metadata.fps:.1f}fps"
            + (f", {self.metadata.frame_count} frames" if self.metadata.frame_count else "")
        )
        return self

    def _read_metadata(self) -> SourceMetadata:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        frame_count: Optional[int] = None
        if self.source_type == "file":
            raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = raw_count if raw_count > 0 else None
        # Webcams report CAP_PROP_FRAME_COUNT as 0/garbage - always
        # None for webcam, deliberately, rather than trusting that.

        return SourceMetadata(
            source_type=self.source_type,
            source=str(self.source),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        """Read and return a single frame (BGR numpy array), or None
        if there are no more frames (end-of-file for a video file, or
        a read failure for a webcam - see module docstring for why
        those aren't distinguished here). Raises VideoSourceError if
        called before open() or after close()."""
        if self._cap is None:
            raise VideoSourceError(
                "VideoSource.read() called before open() - call open() first, "
                "or use this object as a context manager."
            )
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        """Release the underlying capture. Safe to call multiple
        times, and safe to call even if open() was never called or
        already failed."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"VideoSource closed: type={self.source_type}, source={self.source}")

    # ------------------------------------------------------------
    # Iterator protocol - lets Phase 7.3 write `for frame in source:`
    # ------------------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    # ------------------------------------------------------------
    # Context manager protocol - guarantees close() even on exception
    # ------------------------------------------------------------
    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # never suppress exceptions raised inside the `with` block
