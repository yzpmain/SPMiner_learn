from __future__ import annotations

from src.gui.tabs._base import _BaseTab


class TrainTab(_BaseTab):
    MODULE = "src.subgraph_matching.train"
    RUN_TEXT = "Run Training"

    def __init__(self, master, run_command):
        super().__init__(master, run_command)
        self.add_entry("dataset", "syn")
        self.add_entry("model_path", "ckpt/model.pt")
        self.add_entry("batch_size", "64")
        self.add_entry("n_batches", "1000")
        self.add_entry("hidden_dim", "64")
        self.add_entry("n_layers", "8")
        self.add_entry("margin", "0.1")
        self.add_checkbox("node_anchored")
        self.add_extra_args()
        self.add_run_button()
