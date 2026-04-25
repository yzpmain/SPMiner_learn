from __future__ import annotations

import queue
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from src.gui import utils
from src.gui.tabs import AnalysisTab, DataTab, HelpTab, MiningTab, TestTab, TrainTab
from src.gui.worker_thread import Task, WorkerThread


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SPMiner GUI")
        self.geometry("1280x860")
        self.minsize(980, 680)

        self._message_queue: "queue.Queue[dict]" = queue.Queue()
        self._worker = WorkerThread(self._message_queue)
        self._worker.start()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        self.help_tab = HelpTab(self.notebook)
        self.data_tab = DataTab(self.notebook)
        self.train_tab = TrainTab(self.notebook, self.run_command)
        self.test_tab = TestTab(self.notebook, self.run_command)
        self.mining_tab = MiningTab(self.notebook, self.run_command)
        self.analysis_tab = AnalysisTab(self.notebook, self.run_command)

        self.notebook.add(self.help_tab, text="Help")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.train_tab, text="Train")
        self.notebook.add(self.test_tab, text="Test")
        self.notebook.add(self.mining_tab, text="Mining")
        self.notebook.add(self.analysis_tab, text="Analysis")

        log_frame = ttk.LabelFrame(self, text="Runtime Log")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=14)
        self.log_text.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)

        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        ttk.Button(log_frame, text="Cancel Running Task", command=self.cancel_task).grid(
            row=1, column=1, sticky="w", padx=8, pady=(0, 8)
        )
        ttk.Button(log_frame, text="Refresh Data", command=self.refresh_data_views).grid(
            row=1, column=2, sticky="e", padx=8, pady=(0, 8)
        )

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._pump_messages)
        self._log("GUI started. Data previews are read-only.")

    def run_command(self, name: str, command: list[str]) -> None:
        cmd_display = " ".join(command)
        self._log(f"Queue task [{name}] -> {cmd_display}")
        self._worker.submit(Task(name=name, command=command, cwd=utils.repo_root()))

    def clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def cancel_task(self) -> None:
        self._log("Cancelling current task...")
        self._worker.cancel_current()

    def refresh_data_views(self) -> None:
        self.data_tab.refresh_files()
        self.help_tab.refresh()
        self._log("Data and help views refreshed.")

    def _pump_messages(self) -> None:
        try:
            while True:
                msg = self._message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.after(120, self._pump_messages)

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "started":
            self._log(f"[START] {msg['name']}: {' '.join(msg['cmd'])}")
        elif kind == "output":
            self._log(msg.get("line", ""))
        elif kind == "finished":
            self._log(
                f"[END] {msg['name']} return={msg['returncode']} "
                f"elapsed={msg['elapsed']:.2f}s cancelled={msg['cancelled']}"
            )
            self.data_tab.refresh_files()
        elif kind == "error":
            self._log(f"[ERROR] {msg['name']}: {msg['error']}")

    def _log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _on_close(self) -> None:
        self._worker.stop()
        self.destroy()


def run_gui() -> None:
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
