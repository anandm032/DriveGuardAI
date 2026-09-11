"""
DriveGuard AI - Video Source Tests
======================================
Tests detection/video_source.py against REAL video files (generated
on the fly with cv2.VideoWriter, then read back with cv2.VideoCapture
- genuine OpenCV encode/decode, not mocked) and against the webcam
code path WITHOUT requiring a physical camera - a missing/unavailable
camera is treated as an expected, correctly-handled outcome, not a
test failure.

Usage:
    python scripts/test_video_source.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from detection.video_source import SourceMetadata, VideoSource, VideoSourceError


def check(condition: bool, description: str):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {description}")
    assert condition, f"FAILED: {description}"


def make_test_video(path: str, num_frames: int = 5, width: int = 64, height: int = 48, fps: float = 10.0):
    """Write a real, tiny video file - each frame a different solid
    color so frame content is trivially distinguishable in tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Test setup failed: could not open VideoWriter for {path}")
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 40) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ------------------------------------------------------------
# 1. Invalid video source fails clearly
# ------------------------------------------------------------
def test_invalid_source_fails_clearly():
    print("Test 1: invalid video source fails clearly")

    # Nonexistent file path
    vs = VideoSource("/tmp/this_file_definitely_does_not_exist_12345.mp4")
    try:
        vs.open()
        check(False, "should have raised VideoSourceError for a missing file")
    except VideoSourceError as e:
        check("not found" in str(e).lower(), f"clear error message for missing file: {e}")

    # A file that exists but is not a real video (garbage bytes)
    garbage_path = os.path.join(tempfile.gettempdir(), "garbage_not_a_video.mp4")
    with open(garbage_path, "wb") as f:
        f.write(b"this is not a video file, just some bytes")
    vs2 = VideoSource(garbage_path)
    try:
        vs2.open()
        # Some OpenCV builds "open" a garbage file but fail on the
        # first read - accept either failure point as correctly handled.
        frame = vs2.read()
        check(frame is None, "garbage file: opened but correctly produced no readable frame")
        vs2.close()
    except VideoSourceError as e:
        check(True, f"garbage file correctly rejected at open(): {e}")
    os.remove(garbage_path)


# ------------------------------------------------------------
# 2. Video file can be opened
# ------------------------------------------------------------
def test_video_file_can_be_opened():
    print("Test 2: a real video file can be opened, with correct metadata")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.mp4")
        make_test_video(path, num_frames=5, width=64, height=48, fps=10.0)

        vs = VideoSource(path)
        check(vs.source_type == "file", "source_type correctly identified as 'file' before open()")
        vs.open()
        check(vs.is_opened(), "is_opened() reports True after a successful open()")
        check(isinstance(vs.metadata, SourceMetadata), "metadata populated after open()")
        check(vs.metadata.source_type == "file", "metadata.source_type == 'file'")
        check(vs.metadata.width == 64, f"metadata.width == 64 (got {vs.metadata.width})")
        check(vs.metadata.height == 48, f"metadata.height == 48 (got {vs.metadata.height})")
        check(vs.metadata.frame_count == 5, f"metadata.frame_count == 5 (got {vs.metadata.frame_count})")
        vs.close()


# ------------------------------------------------------------
# 3. Frames can be read
# ------------------------------------------------------------
def test_frames_can_be_read():
    print("Test 3: frames can be read, both via read() and iteration")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.mp4")
        make_test_video(path, num_frames=5, width=64, height=48, fps=10.0)

        # Via explicit read()
        with VideoSource(path) as vs:
            frame = vs.read()
            check(frame is not None, "read() returns a real frame, not None")
            check(frame.shape == (48, 64, 3), f"frame shape matches the written video (got {frame.shape})")

        # Via iteration (the interface Phase 7.3 will actually use)
        with VideoSource(path) as vs:
            count = 0
            for frame in vs:
                check(frame is not None, f"iterated frame {count + 1} is not None")
                count += 1
            check(count == 5, f"iteration yielded all 5 written frames (got {count})")


# ------------------------------------------------------------
# 4. End-of-file is handled correctly
# ------------------------------------------------------------
def test_end_of_file_handled_correctly():
    print("Test 4: end-of-file stops iteration cleanly, no error")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.mp4")
        make_test_video(path, num_frames=3, width=64, height=48, fps=10.0)

        with VideoSource(path) as vs:
            frames = list(vs)  # exhausts the iterator via StopIteration
            check(len(frames) == 3, f"exactly 3 frames yielded before StopIteration (got {len(frames)})")

            # Reading again past EOF should keep returning None, not raise
            extra = vs.read()
            check(extra is None, "reading past end-of-file returns None, not an exception")
            extra2 = vs.read()
            check(extra2 is None, "reading past end-of-file repeatedly stays None (no crash)")


# ------------------------------------------------------------
# 5. Resources are released
# ------------------------------------------------------------
def test_resources_are_released():
    print("Test 5: the underlying capture is released correctly")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.mp4")
        make_test_video(path, num_frames=3, width=64, height=48, fps=10.0)

        # Explicit open/close
        vs = VideoSource(path)
        vs.open()
        check(vs.is_opened(), "opened successfully")
        vs.close()
        check(not vs.is_opened(), "is_opened() reports False after close()")
        check(vs._cap is None, "internal capture handle cleared after close()")

        # close() is safe to call multiple times
        vs.close()
        vs.close()
        check(True, "calling close() multiple times does not raise")

        # close() is safe even if open() was never called
        vs3 = VideoSource(path)
        vs3.close()
        check(True, "calling close() before open() does not raise")

        # Context manager releases automatically, even when the body raises
        vs4 = VideoSource(path)
        try:
            with vs4:
                check(vs4.is_opened(), "opened inside the `with` block")
                raise RuntimeError("simulated failure inside the with-block")
        except RuntimeError:
            pass
        check(not vs4.is_opened(), "context manager released the capture even after an exception")

        # read() after close() should raise a clear error, not crash with an AttributeError
        try:
            vs4.read()
            check(False, "read() after close() should raise VideoSourceError")
        except VideoSourceError as e:
            check(True, f"read() after close() raises a clear error: {e}")


# ------------------------------------------------------------
# 5b. Repeated open() is idempotent - no leaked/overwritten capture
# ------------------------------------------------------------
def test_repeated_open_is_idempotent():
    print("Test 5b: calling open() twice does not create/leak a second capture")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.mp4")
        make_test_video(path, num_frames=3, width=64, height=48, fps=10.0)

        vs = VideoSource(path)
        vs.open()
        first_cap = vs._cap
        check(vs.is_opened(), "opened successfully on the first call")

        vs.open()  # second call - must be a no-op, not a second VideoCapture
        check(vs._cap is first_cap,
              "internal capture handle is the SAME object after a second open() call "
              "(no new VideoCapture was constructed, so nothing was leaked)")
        check(vs.is_opened(), "still reports is_opened() == True after the redundant open()")

        # And it must still function normally afterward - the fix
        # shouldn't have left it in some half-working state.
        frame = vs.read()
        check(frame is not None, "reading still works correctly after the redundant open() call")

        vs.close()
        check(not vs.is_opened(), "still closes normally afterward")


# ------------------------------------------------------------
# 6. Webcam source is supported by the interface - no hardware required
# ------------------------------------------------------------
def test_webcam_interface_without_hardware():
    print("Test 6: webcam source is supported, without requiring a physical camera")
    vs = VideoSource(0)
    check(vs.source_type == "webcam", "an int source is correctly identified as 'webcam'")
    check(vs.source == 0, "webcam index stored correctly")

    # This sandbox has no camera, so opening index 0 is EXPECTED to
    # fail - either outcome below is a correct, handled result; the
    # only failure would be an unhandled crash.
    try:
        vs.open()
        check(vs.is_opened(), "a real camera was present and opened successfully")
        frame = vs.read()
        check(frame is not None, "a frame was read from the real camera")
        vs.close()
        print("  [INFO] A physical webcam was detected and used in this environment.")
    except VideoSourceError as e:
        check(True, f"no camera available - failure reported clearly, not a crash: {e}")

    # Same check via the context-manager path
    try:
        with VideoSource(0) as vs2:
            check(vs2.is_opened(), "webcam opened via context manager")
    except VideoSourceError as e:
        check(True, f"context-manager path also fails clearly without hardware: {e}")


def main():
    print("=" * 60)
    print("DriveGuard AI - Video Source Tests")
    print("=" * 60)

    tests = [
        test_invalid_source_fails_clearly,
        test_video_file_can_be_opened,
        test_frames_can_be_read,
        test_end_of_file_handled_correctly,
        test_resources_are_released,
        test_repeated_open_is_idempotent,
        test_webcam_interface_without_hardware,
    ]

    for test_fn in tests:
        test_fn()
        print()

    print("=" * 60)
    print(f"All {len(tests)} test groups passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
