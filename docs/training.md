# Training Engine

Documentation regarding training execution, experimentation, and configurations.

## Architecture components

- `training/trainer.py`: Encapsulates training, validation, and epoch loops.
- `training/callbacks.py`: Support for EarlyStopping, WandbLogging, and Checkpointing.
- `training/losses.py` & `training/optimizers.py`: Customizable and modular optimization layers.
- `training/experiment.py`: Manages reproducible seeds, setting up metrics directories, and reading parameters.

## Logging & Versioning
All configurations are defined in `configs/training.yaml` and output logs are routed directly to `logs/training/`. Checkpoints and metrics are logged to unique runs under `experiments/exp{NNN}/`.
