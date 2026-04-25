from __future__ import annotations

from src.gui.tabs._base import _BaseTab


class TestTab(_BaseTab):
    MODULE = "src.subgraph_matching.test"
    RUN_TEXT = "Run Evaluation"

    def __init__(self, master, run_command):
        super().__init__(master, run_command)
        self.add_entry("dataset", "syn")
        self.add_entry("model_path", "ckpt/model.pt")
        self.add_entry("batch_size", "64")
        self.add_entry("val_size", "256")
        self.add_extra_args()
        self.add_run_button()
