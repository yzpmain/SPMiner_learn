from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Callable

from src.gui import utils

RunCallback = Callable[[str, list[str]], None]


class _BaseTab(ttk.Frame):
    """CLI 命令标签页基类，封装通用的表单布局与命令构建逻辑。"""

    MODULE: str = ""
    RUN_TEXT: str = "Run"
    RUN_NAME: str = ""

    def __init__(self, master: tk.Misc, run_command: RunCallback) -> None:
        super().__init__(master)
        self._run_command = run_command
        self.columnconfigure(1, weight=1)
        self._extra_var = tk.StringVar(value="")
        self._row = 0
        self._fields: list[tuple[str, str, tk.StringVar | tk.BooleanVar]] = []

    # -- field builders -------------------------------------------------------

    def add_entry(self, name: str, default: str = "") -> tk.StringVar:
        var = tk.StringVar(value=default)
        self._fields.append(("entry", name, var))
        ttk.Label(self, text=name).grid(row=self._row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self, textvariable=var).grid(row=self._row, column=1, sticky="ew", padx=8, pady=4)
        self._row += 1
        return var

    def add_combobox(self, name: str, default: str, values: list[str]) -> tk.StringVar:
        var = tk.StringVar(value=default)
        self._fields.append(("combobox", name, var))
        ttk.Label(self, text=name).grid(row=self._row, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(self, textvariable=var, state="readonly", values=values).grid(
            row=self._row, column=1, sticky="ew", padx=8, pady=4
        )
        self._row += 1
        return var

    def add_checkbox(self, name: str, default: bool = False) -> tk.BooleanVar:
        var = tk.BooleanVar(value=default)
        self._fields.append(("checkbox", name, var))
        ttk.Checkbutton(self, text=name, variable=var).grid(
            row=self._row, column=0, columnspan=2, sticky="w", padx=8, pady=4
        )
        self._row += 1
        return var

    def add_extra_args(self) -> None:
        ttk.Label(self, text="extra args").grid(row=self._row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self, textvariable=self._extra_var).grid(row=self._row, column=1, sticky="ew", padx=8, pady=4)
        self._row += 1

    def add_run_button(self, text: str | None = None) -> None:
        ttk.Button(self, text=text or self.RUN_TEXT, command=self.run).grid(
            row=self._row, column=0, columnspan=2, pady=10
        )

    # -- command building -----------------------------------------------------

    def build_command(self) -> list[str]:
        cmd = [sys.executable, "-m", self.MODULE]
        for _kind, name, var in self._fields:
            if isinstance(var, tk.BooleanVar):
                if var.get():
                    cmd.append(f"--{name}")
            else:
                val = var.get().strip()
                if val:
                    cmd.extend([f"--{name}", val])
        extra = self._extra_var.get().strip()
        if extra:
            cmd.extend(utils.split_extra_args(extra))
        return cmd

    def run(self) -> None:
        name = self.RUN_NAME or self.MODULE.rsplit(".", 1)[-1]
        self._run_command(name, self.build_command())
