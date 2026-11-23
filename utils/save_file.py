import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

import numpy as np


def format_video_saved_message(
    path: str,
    n_frames: int,
    width: int,
    height: int,
    fps: float,
) -> str:
    return (
        f"🎬 Video is saved to `{path}`, containing "
        f"\033[94m{n_frames}\033[0m frames at {width}×{height} "
        f"resolution and {fps} FPS."
    )


class VideoStreamWriter:
    """Stream frames to an mp4 through a persistent ffmpeg pipe.

    Keeps a single ffmpeg process alive and flushes each frame to disk as soon
    as it arrives, so an episode's frames are never all held in memory at once.
    Use ``append`` per frame, then ``close`` to finalize a valid file or
    ``abort`` to discard the partial output.

    Frames are handed to a writer thread through a bounded queue so the caller
    does not block while ffmpeg encodes; ``close`` drains every queued frame.
    """

    _QUEUE_FRAMES = 64

    def __init__(
        self,
        out_path: str,
        height: int,
        width: int,
        channels: int,
        fps: float = 30.0,
        is_rgb: bool = True,
    ) -> None:
        if channels == 3:
            pixel_format = "rgb24" if is_rgb else "bgr24"
        elif channels == 4:
            pixel_format = "rgba"
        else:
            raise ValueError(f"Unsupported channel count for video: {channels}")
        self.out_path = out_path
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.n_frames = 0
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._drain, name=f"video:{os.path.basename(out_path)}", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        stdin = self.proc.stdin
        while True:
            data = self._queue.get()
            if data is None:
                return
            if self._error is not None:
                continue  # keep consuming so producers never block on a dead pipe
            try:
                stdin.write(data)
            except BaseException as e:  # noqa: BLE001 - re-raised on the caller thread
                self._error = e

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def append(self, frame: np.ndarray) -> None:
        if self.proc is None:
            raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
        if self._error is not None:
            raise OSError(f"ffmpeg pipe failed for `{self.out_path}`.") from self._error
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} does not match writer ({self.height}x{self.width}x{self.channels})."
            )
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        # tobytes() copies, so the caller may reuse its buffer immediately.
        self._queue.put(frame.tobytes())
        self.n_frames += 1

    def close(self, *, announce: bool = True) -> None:
        if self.proc is None:
            return
        try:
            self._stop_thread()
            if self._error is not None:
                raise OSError(f"ffmpeg pipe failed for `{self.out_path}`.") from self._error
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise OSError(f"ffmpeg failed while finalizing `{self.out_path}`.")
        finally:
            self.proc = None
        if announce:
            print(
                format_video_saved_message(
                    self.out_path,
                    self.n_frames,
                    self.width,
                    self.height,
                    self.fps,
                )
            )

    def abort(self) -> None:
        """Kill the ffmpeg process and remove the partial output file."""
        if self.proc is not None:
            try:
                # Kill first so a writer thread blocked on a full pipe gets EPIPE and exits.
                self.proc.kill()
            except Exception:
                pass
            try:
                self._stop_thread()
            except Exception:
                pass
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait()
            except Exception:
                pass
            self.proc = None
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
        except Exception:
            pass


def save_json(
    data: Any,
    path: str | os.PathLike,
    overwrite: bool = True,
    make_dirs: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    p = Path(path)

    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists and overwrite=False")

    tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                data,
                f,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
            )
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise
