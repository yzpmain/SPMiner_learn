from __future__ import annotations

from dataclasses import dataclass
import queue
import subprocess
import threading
import time
from pathlib import Path


@dataclass
class Task:
    name: str
    command: list[str]
    cwd: Path


class WorkerThread(threading.Thread):
    def __init__(self, message_queue: "queue.Queue[dict]") -> None:
        super().__init__(daemon=True)
        self._message_queue = message_queue
        self._task_queue: "queue.Queue[Task]" = queue.Queue()
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._proc_lock = threading.Lock()
        self._current_proc: subprocess.Popen[str] | None = None

    def submit(self, task: Task) -> None:
        self._task_queue.put(task)

    def stop(self) -> None:
        self._stop_event.set()
        self.cancel_current()
        self._task_queue.put(Task(name="__stop__", command=[], cwd=Path.cwd()))

    def cancel_current(self) -> None:
        self._cancel_event.set()
        with self._proc_lock:
            proc = self._current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            task = self._task_queue.get()
            if task.name == "__stop__":
                return
            self._cancel_event.clear()
            self._run_task(task)

    def _run_task(self, task: Task) -> None:
        started = time.time()
        self._message_queue.put({"type": "started", "name": task.name, "cmd": task.command})

        try:
            proc = subprocess.Popen(
                task.command,
                cwd=str(task.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            with self._proc_lock:
                self._current_proc = proc
        except Exception as exc:
            self._message_queue.put({"type": "error", "name": task.name, "error": str(exc)})
            return

        assert proc.stdout is not None

        try:
            for line in proc.stdout:
                if self._cancel_event.is_set() and proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                self._message_queue.put({"type": "output", "name": task.name, "line": line.rstrip("\n")})
        finally:
            proc.stdout.close()

        code = proc.wait()
        with self._proc_lock:
            self._current_proc = None
        elapsed = time.time() - started
        self._message_queue.put(
            {
                "type": "finished",
                "name": task.name,
                "returncode": code,
                "elapsed": elapsed,
                "cancelled": self._cancel_event.is_set(),
            }
        )
