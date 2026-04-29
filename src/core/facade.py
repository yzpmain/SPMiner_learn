"""High-level core facade.

Provides a stable service-style API while keeping CLI compatibility.
"""

from __future__ import annotations

from src.core import artifacts
from src.core import dataset_provider
from src.core import model_factory
from src.core import runtime_context


class CoreFacade:
    @staticmethod
    def setup_runtime(args):
        return runtime_context.build_runtime_context(args)

    @staticmethod
    def build_model(args, *, for_inference: bool = False, load_weights: bool = False):
        model = model_factory.build_from_args(args, for_inference=for_inference)
        if load_weights:
            model_factory.load_state_dict_if_needed(model, getattr(args, "model_path", None))
        return model

    @staticmethod
    def make_matching_data_source(args):
        return dataset_provider.make_matching_data_source(args)

    @staticmethod
    def load_stage_dataset(dataset: str, stage: str):
        return dataset_provider.load_for_stage(dataset, stage)

    @staticmethod
    def stage_artifact_dir(args, task: str, dataset: str | None = None):
        return artifacts.task_output_dir(args, task, dataset)

    @staticmethod
    def choose_output_path(args, cli_path: str, *, default_cli_path: str, suggested_default_path):
        return artifacts.choose_cli_output_path(
            args,
            cli_path,
            default_cli_path=default_cli_path,
            suggested_default_path=suggested_default_path,
        )

    @staticmethod
    def write_manifest(path, args, outputs: dict, **extra):
        return artifacts.write_manifest(path, args, outputs, **extra)
