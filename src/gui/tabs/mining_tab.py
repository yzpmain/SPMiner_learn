from __future__ import annotations

from src.gui.tabs._base import _BaseTab


class MiningTab(_BaseTab):
    MODULE = "src.subgraph_mining.decoder"
    RUN_NAME = "mining"
    RUN_TEXT = "Run Mining"

    def __init__(self, master, run_command):
        super().__init__(master, run_command)
        self.add_entry("dataset", "facebook")
        self.add_entry("model_path", "ckpt/model.pt")
        self.add_entry("out_path", "results/out-patterns.p")
        self.add_combobox("search_strategy", "greedy", ["greedy", "mcts"])
        self.add_entry("min_pattern_size", "5")
        self.add_entry("max_pattern_size", "10")
        self.add_entry("n_trials", "100")
        self.add_extra_args()
        self.add_run_button()
