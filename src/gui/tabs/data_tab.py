from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from src.gui import utils

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None


class DataTab(ttk.Frame):
    """文件浏览标签页：左侧文件列表，右侧文本/图像双面板预览。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._files: list[Path] = []
        self._preview_image = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_file_panel()
        self._build_preview_panel()
        self.refresh_files()

    # -- file panel (left) ----------------------------------------------------

    def _build_file_panel(self) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsw", padx=8, pady=8)

        toolbar = ttk.Frame(left)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(toolbar, text="Type").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="All")
        filter_box = ttk.Combobox(
            toolbar,
            textvariable=self.filter_var,
            state="readonly",
            width=10,
            values=["All", "Dataset", "Text", "Image", "JSON"],
        )
        filter_box.pack(side=tk.LEFT, padx=6)
        filter_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_files())

        ttk.Button(toolbar, text="Refresh", command=self.refresh_files).pack(side=tk.LEFT)

        self.file_list = tk.Listbox(left, width=44, height=30)
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self.on_select)

    # -- preview panel (right, notebook with text + image tabs) ---------------

    def _build_preview_panel(self) -> None:
        right = ttk.Notebook(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)

        self._text_frame = ttk.Frame(right)
        self._image_frame = ttk.Frame(right)
        self._text_frame.columnconfigure(0, weight=1)
        self._text_frame.rowconfigure(0, weight=1)
        self._image_frame.columnconfigure(0, weight=1)
        self._image_frame.rowconfigure(0, weight=1)

        self.preview_text = ScrolledText(self._text_frame, wrap=tk.NONE)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        self.preview_text.configure(state=tk.DISABLED)

        self.preview_canvas = tk.Canvas(self._image_frame, bg="#f3f3f3", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        right.add(self._text_frame, text="Text Preview")
        right.add(self._image_frame, text="Image Preview")

    # -- file operations ------------------------------------------------------

    def refresh_files(self) -> None:
        root = utils.repo_root()
        files = utils.list_files(
            [root / "data", root / "plots", root / "results", root / "result"],
            recursive=True,
        )

        ftype = self.filter_var.get()
        if ftype == "Dataset":
            files = [p for p in files if p.parent.name == "data"]
        elif ftype == "Text":
            files = [p for p in files if utils.is_text(p)]
        elif ftype == "Image":
            files = [p for p in files if utils.is_image(p)]
        elif ftype == "JSON":
            files = [p for p in files if p.suffix.lower() == ".json"]

        self._files = files
        self.file_list.delete(0, tk.END)
        for p in files:
            self.file_list.insert(tk.END, utils.norm_path(p))

        self._set_text("Select a file from the left list.\n\nRead-only preview only.")
        self.preview_canvas.delete("all")

    def on_select(self, _event: tk.Event) -> None:
        idxs = self.file_list.curselection()
        if not idxs:
            return
        path = self._files[idxs[0]]

        if utils.is_image(path):
            self._show_image(path)
            self._set_text(f"Image selected: {utils.norm_path(path)}")
        else:
            preview = utils.read_text_preview(path, max_lines=160)
            self._set_text(preview)
            self.preview_canvas.delete("all")

    # -- preview rendering ----------------------------------------------------

    def _set_text(self, value: str) -> None:
        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, value)
        self.preview_text.configure(state=tk.DISABLED)

    def _show_image(self, path: Path) -> None:
        self.preview_canvas.delete("all")

        if Image is not None and ImageTk is not None:
            try:
                img = Image.open(path)
                img.thumbnail((900, 650))
                self._preview_image = ImageTk.PhotoImage(img)
                self.preview_canvas.create_image(8, 8, image=self._preview_image, anchor=tk.NW)
                return
            except Exception as exc:
                self.preview_canvas.create_text(10, 10, text=f"Image preview failed: {exc}", anchor=tk.NW)
                return

        try:
            self._preview_image = tk.PhotoImage(file=str(path))
            self.preview_canvas.create_image(8, 8, image=self._preview_image, anchor=tk.NW)
        except tk.TclError:
            self.preview_canvas.create_text(
                10,
                10,
                anchor=tk.NW,
                text=(
                    "Image preview requires Pillow for this format.\n"
                    "Install: pip install Pillow"
                ),
            )
