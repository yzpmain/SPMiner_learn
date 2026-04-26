# AGENTS.md

This repository is a two-stage neural subgraph learning project: train a subgraph matching encoder first, then reuse it for frequent subgraph mining. Keep guidance here short and operational; link to existing docs instead of duplicating them.

## Use These Docs First

- [README.md](README.md) for the full project overview, workflows, and parameter tables.
- [CLAUDE.md](CLAUDE.md) for the fastest common commands.
- [environment.yml](environment.yml) for the expected Python and package versions.

## Working Rules

- Run project code from the repository root with `python -m ...`; direct script execution can fail because of the src layout.
- Treat [data/](data/), [ckpt/](ckpt/), [results/](results/), [plots/](plots/), and [runlogs/](runlogs/) as generated or large-output locations unless a task explicitly asks to modify them.
- Prefer the existing argparse/config patterns in [src/subgraph_matching/](src/subgraph_matching/) and [src/subgraph_mining/](src/subgraph_mining/) when adding or changing CLI behavior.
- Keep changes focused on [src/](src/) and [tests/](tests/); avoid unrelated refactors.
- Preserve the current two-stage flow: matching before mining.

## Entry Points

- [src/subgraph_matching/train.py](src/subgraph_matching/train.py) trains the encoder.
- [src/subgraph_matching/test.py](src/subgraph_matching/test.py) evaluates the encoder.
- [src/subgraph_mining/decoder.py](src/subgraph_mining/decoder.py) runs mining.
- [src/gui/main_window.py](src/gui/main_window.py) launches the GUI.
- [src/analyze/](src/analyze/) contains post-processing and count/analysis scripts.

## Environment and Testing

- Target Python 3.10 with the conda environment defined in [environment.yml](environment.yml).
- Be careful with PyTorch Geometric and DeepSNAP versions; mismatches are a common source of setup issues on Windows.
- Use `pytest` from the repository root. The test discovery configuration is in [pyproject.toml](pyproject.toml), and the main suite lives under [tests/](tests/).
- When validating a change, run the smallest relevant test scope first.

## Logging and Outputs

- Runtime logs are written through [src/logger.py](src/logger.py) into timestamped folders under [runlogs/](runlogs/).
- If a task creates or updates artifacts, prefer writing to an existing output directory and document the path in the task result.

## When Extending the Codebase

- If you add new user-facing commands or workflows, update [README.md](README.md) and, if useful, [CLAUDE.md](CLAUDE.md) to keep the quick-start path current.
- If you discover a recurring setup or execution pitfall, add a short note here rather than repeating it in multiple places.
