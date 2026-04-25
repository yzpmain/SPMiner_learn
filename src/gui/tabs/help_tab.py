from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from src.gui import utils


class HelpTab(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side=tk.LEFT)

        self.text = ScrolledText(self, wrap=tk.WORD, height=30)
        self.text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.text.configure(state=tk.DISABLED)

        self.refresh()

    def refresh(self) -> None:
        lines: list[str] = []
        lines.append("Project Features (auto-discovered)")
        lines.append("=" * 60)

        modules = utils.discover_cli_modules()
        if modules:
            for mod in modules:
                lines.append(f"- {mod}")
        else:
            lines.append("No CLI modules discovered.")

        lines.append("")
        lines.append("README Sections")
        lines.append("=" * 60)
        for title, body in utils.extract_readme_sections():
            lines.append(f"\n[{title}]\n{body}\n")

        content = "\n".join(lines)
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, content)
        self.text.configure(state=tk.DISABLED)
