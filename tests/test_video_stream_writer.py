"""VideoStreamWriter hands frames to a writer thread without losing any."""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from utils.save_file import VideoStreamWriter


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
class VideoStreamWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def frames(self, count):
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8) for _ in range(count)]

    def count_frames(self, path):
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip())

    def test_close_flushes_every_queued_frame(self):
        path = os.path.join(self.tmp, "out.mp4")
        writer = VideoStreamWriter(path, 48, 64, 3, fps=25)
        count = VideoStreamWriter._QUEUE_FRAMES * 3  # more than the queue holds
        for frame in self.frames(count):
            writer.append(frame)
        writer.close(announce=False)
        self.assertEqual(writer.n_frames, count)
        self.assertEqual(self.count_frames(path), count)

    def test_caller_may_reuse_its_buffer(self):
        path = os.path.join(self.tmp, "reuse.mp4")
        writer = VideoStreamWriter(path, 48, 64, 3, fps=25)
        buffer = np.zeros((48, 64, 3), dtype=np.uint8)
        for value in range(10):
            buffer[:] = value * 20
            writer.append(buffer)
        writer.close(announce=False)
        self.assertEqual(self.count_frames(path), 10)

    def test_abort_with_full_queue_removes_output(self):
        path = os.path.join(self.tmp, "abort.mp4")
        writer = VideoStreamWriter(path, 48, 64, 3, fps=25)
        for frame in self.frames(VideoStreamWriter._QUEUE_FRAMES + 5):
            writer.append(frame)
        writer.abort()
        self.assertIsNone(writer.proc)
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
