# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository (TFG - Training-Free Guidance) implements various guidance methods for diffusion models, including temporal guidance and energy-based guidance for different modalities (image, molecule, audio).

## Architecture

- **Entry Point**: `main.py` initializes the configuration, network, guider, and pipeline, then runs sampling and evaluation.
- **Pipeline**: `pipeline.py` defines `BasePipeline` which orchestrates the generation process, applying guidance during sampling.
- **Configuration**: `utils/configs.py` uses `HfArgumentParser` to handle explicit CLI arguments. `utils/env_utils.py` contains hardcoded paths that **must check/update** before running.
- **Core Components**:
  - **Diffusion**: `diffusion/` contains sampler wrappers (`ImageSampler`, `StableDiffusionSampler`, etc.).
  - **Methods**: `methods/` contains guidance implementations (e.g., `TFGGuidance`, `MPGDGuidance`, `ClassifierGuidance`).
  - **Tasks**: `tasks/` defines specific task logic and energy functions (e.g., `TangentPointEnergyGuidance` for wireframes).
  - **Evaluations**: `evaluations/` handles metric calculation (FID, etc.).

## Development & Usage

### Setup
1. **Critical Module Path**: The code assumes it is run from the root directory. Ensure `sys.path` allows importing `utils`, `diffusion`, etc.
2. **Environment Paths**: Check `utils/env_utils.py` and strictly follow the comment `# PLEASE CHANGE THEM TO YOUR OWN PATHs BEFORE RUNNING !!`.

### Running Generation
The main script is `main.py`. Arguments are defined in `utils/configs.py`.

```bash
# Example: Run image generation with default settings
python main.py --data_type image --dataset cifar10

# Example: Run with TFG guidance
python main.py --guidance_name tfg --rho 1.0 --mu 1.0 --sigma 0.01
```

### Key Arguments
See `utils/configs.py` for all options.
- `--data_type`: `image`, `molecule`, `text2image`, `audio`.
- `--guidance_name`: `no`, `mpgd`, `ugd`, `freedom`, `dps`, `lgd`, `tfg`, `cg`.
- `--guidance_strength`: Float value for guidance scale.
- `--seed`: Random seed (default: 42).

### Testing & Evaluation
- There is no dedicated test runner (e.g., pytest) visible.
- Evaluation happens automatically at the end of `main.py` if an evaluator is available for the data type.
- Metrics are logged to `logs/` directory structure defined in `utils/utils.py`.

## Style & Conventions
- **Imports**: Absolute imports from project root (e.g., `from utils.utils import ...`).
- **Typos**: Be aware of typos in variable names (e.g., `partiprompts` vs `PARTIPROMPOTS_PATH` in `env_utils.py`).
- **Error Handling**: `NotImplementedError` is frequently used for unsupported configurations.
