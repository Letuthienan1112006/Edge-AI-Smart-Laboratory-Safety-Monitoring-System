from __future__ import annotations

import json

import cv2
import numpy as np

from labsafe.io.recording import AsyncVideoRecorder


def test_async_video_recorder_writes_playable_video(tmp_path):
    path = tmp_path / "annotated.mp4"
    recorder = AsyncVideoRecorder(path, (96, 64), fps=10, queue_size=8)
    for i in range(6):
        frame = np.full((64, 96, 3), i * 30, dtype=np.uint8)
        recorder.write(frame)
    stats = recorder.close()

    assert stats["error"] is None
    assert stats["frames_submitted"] == 6
    assert stats["frames_written"] >= 1
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (64, 96)


def test_async_video_recorder_stats_are_json_serializable(tmp_path):
    recorder = AsyncVideoRecorder(tmp_path / "x.mp4", (32, 24), fps=5)
    recorder.write(np.zeros((24, 32, 3), dtype=np.uint8))
    json.dumps(recorder.close())
