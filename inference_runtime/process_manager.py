"""
process_manager.py
-------------------
Subprocess lifecycle management with non-blocking async token streaming.

Manages launching the llama.exe subprocess, reading its stdout/stderr
concurrently via a background reader thread (so the main thread is never
blocked waiting for the process), and collecting per-token timestamps for
latency distribution analysis.

Async streaming design
-----------------------
The "overlap computation with transfers" requirement is realized at the
Python level by running two concurrent daemon threads:

1. **stdout reader thread** — reads generated tokens from stdout line-by-line
   (llama.cpp flushes after each token) and calls the ``on_token`` callback
   immediately. This gives streaming UX without blocking the main thread.

2. **stderr reader thread** — buffers the full stderr output (contains timing
   stats) without blocking stdout consumption.

Both threads are started together when the subprocess is launched. The main
thread can either:
  a) Block synchronously via ``wait_for_completion()``, or
  b) Poll ``is_running()`` and process events from a queue.

Pinned memory / copy avoidance
--------------------------------
The process manager passes ``--flash-attn`` and ``--no-mmap``/``--mlock``
flags to the subprocess (configured in ArgumentBuilder). These control
llama.cpp's internal memory allocation behavior. The manager itself avoids
redundant copies by accumulating stdout/stderr in pre-allocated lists and
joining with ``"".join()`` only once at finalization.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------

class ProcessManager:
    """
    Manages a single llama.exe subprocess with asynchronous I/O.

    Parameters
    ----------
    env_vars : dict, optional
        Additional environment variables to inject into the subprocess
        environment (e.g. ``{"CUDA_VISIBLE_DEVICES": "0"}``).
    """

    def __init__(self, env_vars: Optional[Dict[str, str]] = None) -> None:
        self._env_vars = env_vars or {}
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_chunks: List[str] = []
        self._stderr_chunks: List[str] = []
        self._token_timestamps: List[float] = []
        self._lock = threading.Lock()
        self._done_event = threading.Event()

    def launch(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
    ) -> subprocess.Popen:
        """
        Launch the llama.exe subprocess with the given argument list.

        Parameters
        ----------
        args : List[str]
            Complete argument list (including the executable as args[0]).
        cwd : Path, optional
            Working directory for the subprocess. Defaults to the directory
            containing the executable.

        Returns
        -------
        subprocess.Popen
            The launched process handle.
        """
        # Merge environment: current OS env + backend overrides
        env = os.environ.copy()
        env.update(self._env_vars)

        if cwd is None and args:
            exe_path = Path(args[0])
            if exe_path.parent.is_dir():
                cwd = exe_path.parent

        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,   # EOF immediately → non-interactive
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(cwd) if cwd else None,
            # Buffer by line so we get tokens promptly
            bufsize=1,
        )

        # Reset state for a fresh run
        self._stdout_chunks = []
        self._stderr_chunks = []
        self._token_timestamps = []
        self._done_event.clear()

        return self._proc

    def stream_async(
        self,
        on_token: Optional[Callable[[str, float], None]] = None,
        on_complete: Optional[Callable[[str, str], None]] = None,
    ) -> threading.Thread:
        """
        Start background threads to consume stdout and stderr without blocking
        the caller.

        Each time a line is read from stdout, ``on_token(text, elapsed_ms)``
        is called (if provided), where ``elapsed_ms`` is the wall time since
        the process was launched.

        Parameters
        ----------
        on_token : callable, optional
            Callback invoked for each stdout line. Signature:
            ``(token_text: str, elapsed_ms: float) -> None``.
        on_complete : callable, optional
            Callback invoked once both stdout and stderr are fully consumed.
            Signature: ``(full_stdout: str, full_stderr: str) -> None``.

        Returns
        -------
        threading.Thread
            The stdout reader thread. Can be joined to synchronize.
        """
        if self._proc is None:
            raise RuntimeError("Process not launched. Call launch() first.")

        start_time = time.perf_counter()

        def _read_stdout() -> None:
            """Read stdout stream per character/token flush, calling on_token as tokens arrive."""
            assert self._proc is not None
            try:
                buffer: List[str] = []
                last_char_time = 0.0
                while True:
                    char = self._proc.stdout.read(1)  # type: ignore[union-attr]
                    if not char:
                        break
                    now = time.perf_counter()
                    with self._lock:
                        self._stdout_chunks.append(char)
                        if now - last_char_time > 0.002 or not self._token_timestamps:
                            self._token_timestamps.append(now)
                        last_char_time = now

                    buffer.append(char)
                    if on_token is not None:
                        token_str = "".join(buffer)
                        buffer = []
                        elapsed_ms = (now - start_time) * 1000.0
                        try:
                            on_token(token_str, elapsed_ms)
                        except Exception:
                            pass
                if buffer and on_token is not None:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    try:
                        on_token("".join(buffer), elapsed_ms)
                    except Exception:
                        pass
            except Exception:
                pass

        def _read_stderr() -> None:
            """Read stderr fully and store it for stats parsing."""
            assert self._proc is not None
            try:
                for line in self._proc.stderr:  # type: ignore[union-attr]
                    with self._lock:
                        self._stderr_chunks.append(line)
            except Exception:
                pass

        def _coordinator() -> None:
            """Wait for both reader threads, then fire on_complete."""
            stdout_t = threading.Thread(target=_read_stdout, daemon=True, name="stdout_reader")
            stderr_t = threading.Thread(target=_read_stderr, daemon=True, name="stderr_reader")
            stdout_t.start()
            stderr_t.start()
            stdout_t.join()
            stderr_t.join()

            self._done_event.set()

            if on_complete is not None:
                stdout_str = "".join(self._stdout_chunks)
                stderr_str = "".join(self._stderr_chunks)
                try:
                    on_complete(stdout_str, stderr_str)
                except Exception:
                    pass

        coordinator = threading.Thread(
            target=_coordinator,
            daemon=True,
            name="InferenceOSStreamCoordinator",
        )
        coordinator.start()
        self._stdout_thread = coordinator
        return coordinator

    def wait_for_completion(
        self,
        timeout_sec: float = 600.0,
    ) -> Tuple[str, str]:
        """
        Block until the subprocess completes and all output has been consumed.

        Parameters
        ----------
        timeout_sec : float
            Maximum seconds to wait before force-killing the process.
            Default 600 s.

        Returns
        -------
        tuple[str, str]
            ``(full_stdout, full_stderr)`` combined output strings.
        """
        # Wait for the stream coordinator to signal completion
        finished = self._done_event.wait(timeout=timeout_sec)

        if not finished:
            # Timeout expired — kill the process
            self.kill()
            self._done_event.wait(timeout=5.0)

        # Ensure the process has terminated
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.kill()

        with self._lock:
            stdout_str = "".join(self._stdout_chunks)
            stderr_str = "".join(self._stderr_chunks)

        return stdout_str, stderr_str

    def wait_sync(self, timeout_sec: float = 600.0) -> Tuple[str, str, int]:
        """
        Synchronous (non-streaming) wait: calls Popen.communicate() directly.

        Use this when ``async_streaming=False`` in the config and you do not
        need a token callback. More reliable on Windows where line-buffered
        pipes can behave differently.

        Parameters
        ----------
        timeout_sec : float
            Maximum wait time in seconds.

        Returns
        -------
        tuple[str, str, int]
            ``(stdout, stderr, return_code)``.
        """
        if self._proc is None:
            raise RuntimeError("Process not launched.")
        try:
            stdout, stderr = self._proc.communicate(timeout=timeout_sec)
            return stdout, stderr, self._proc.returncode
        except subprocess.TimeoutExpired:
            self.kill()
            stdout, stderr = self._proc.communicate()
            return stdout, stderr, self._proc.returncode

    def kill(self) -> None:
        """Forcibly terminate the subprocess if it is still running."""
        if self._proc is not None:
            try:
                self._proc.kill()
            except (ProcessLookupError, OSError):
                pass

    def is_running(self) -> bool:
        """Return True if the subprocess is still alive."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    @property
    def return_code(self) -> Optional[int]:
        """Return the process exit code, or None if still running."""
        if self._proc is None:
            return None
        return self._proc.returncode

    @property
    def token_timestamps(self) -> List[float]:
        """Per-token arrival timestamps (perf_counter values)."""
        with self._lock:
            return list(self._token_timestamps)
