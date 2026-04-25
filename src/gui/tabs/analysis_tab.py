from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

from src.gui import utils
from src.gui.tabs._base import RunCallback


class AnalysisTab(ttk.Frame):
    """分析标签页：包含 Count Patterns 和 Analyze Count Outputs 两个子表单。"""

    def __init__(self, master: tk.Misc, run_command: RunCallback) -> None:
        super().__init__(master)
        self._run_command = run_command

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._build_count_frame()
        self._build_summary_frame()

    # -- sub-form helpers -----------------------------------------------------

    @staticmethod
    def _add_entry(parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=4)

    @staticmethod
    def _add_extra(parent: ttk.Frame, row: int, var: tk.StringVar) -> None:
        ttk.Label(parent, text="extra args").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=4)

    @staticmethod
    def _add_button(parent: ttk.Frame, row: int, text: str, command) -> None:
        ttk.Button(parent, text=text, command=command).grid(
            row=row, column=0, columnspan=2, pady=8
        )

    # -- Count Patterns sub-form ----------------------------------------------

    def _build_count_frame(self) -> None:
        frame = ttk.LabelFrame(self, text="Count Patterns")
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        frame.columnconfigure(1, weight=1)

        self.dataset_var = tk.StringVar(value="facebook")
        self.queries_path_var = tk.StringVar(value="results/out-patterns.p")
        self.count_out_var = tk.StringVar(value="results/counts.json")
        self.n_workers_var = tk.StringVar(value="4")
        self.count_method_var = tk.StringVar(value="bin")
        self.node_anchored_var = tk.BooleanVar(value=False)
        self.use_orbitsi_var = tk.BooleanVar(value=False)
        self.count_extra_var = tk.StringVar(value="")

        row = 0
        for label, var in [
            ("dataset", self.dataset_var),
            ("queries_path", self.queries_path_var),
            ("out_path", self.count_out_var),
            ("n_workers", self.n_workers_var),
        ]:
            self._add_entry(frame, row, label, var)
            row += 1

        ttk.Label(frame, text="count_method").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.count_method_var,
            state="readonly",
            values=["bin", "freq"],
        ).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        row += 1

        ttk.Checkbutton(frame, text="node_anchored", variable=self.node_anchored_var).grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(frame, text="use_orbitsi", variable=self.use_orbitsi_var).grid(
            row=row, column=1, sticky="w", padx=6, pady=4
        )
        row += 1

        self._add_extra(frame, row, self.count_extra_var)
        row += 1
        self._add_button(frame, row, "Run Count", self.run_count)

    # -- Analyze Count Outputs sub-form ---------------------------------------

    def _build_summary_frame(self) -> None:
        frame = ttk.LabelFrame(self, text="Analyze Count Outputs")
        frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        frame.columnconfigure(1, weight=1)

        self.counts_path_var = tk.StringVar(value="results")
        self.analysis_out_var = tk.StringVar(value="results/analysis.csv")
        self.summary_extra_var = tk.StringVar(value="")

        row = 0
        for label, var in [
            ("counts_path", self.counts_path_var),
            ("out_path", self.analysis_out_var),
        ]:
            self._add_entry(frame, row, label, var)
            row += 1

        self._add_extra(frame, row, self.summary_extra_var)
        row += 1
        self._add_button(frame, row, "Run Analyze", self.run_summary)

    # -- run methods ----------------------------------------------------------

    def run_count(self) -> None:
        cmd = [sys.executable, "-m", "src.analyze.count_patterns"]
        utils.append_arg(cmd, "dataset", self.dataset_var.get())
        utils.append_arg(cmd, "queries_path", self.queries_path_var.get())
        utils.append_arg(cmd, "out_path", self.count_out_var.get())
        utils.append_arg(cmd, "n_workers", self.n_workers_var.get())
        utils.append_arg(cmd, "count_method", self.count_method_var.get())
        utils.append_flag(cmd, "node_anchored", self.node_anchored_var.get())
        utils.append_flag(cmd, "use_orbitsi", self.use_orbitsi_var.get())
        cmd.extend(utils.split_extra_args(self.count_extra_var.get()))
        self._run_command("analysis.count", cmd)

    def run_summary(self) -> None:
        cmd = [sys.executable, "-m", "src.analyze.analyze_pattern_counts"]
        utils.append_arg(cmd, "counts_path", self.counts_path_var.get())
        utils.append_arg(cmd, "out_path", self.analysis_out_var.get())
        cmd.extend(utils.split_extra_args(self.summary_extra_var.get()))
        self._run_command("analysis.summary", cmd)
