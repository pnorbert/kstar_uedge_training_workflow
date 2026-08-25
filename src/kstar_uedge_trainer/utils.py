from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration shared by the testing-set and training workflows."""

    acx: Path
    uedge_campaign_rootdir: Path
    random_state: int
    vae_epochs: int
    mlp_epochs: int


def read_config(config_file: str | Path = "workflow_config.yaml") -> WorkflowConfig:
    """Read and validate workflow settings from a YAML configuration file."""
    config_path = Path(config_file).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")

    try:
        with config_path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc

    if not isinstance(values, dict):
        raise ValueError(f"Configuration file {config_path} must contain a YAML mapping")

    required = {"ACX", "UEDGE_CAMPAIGN_ROOTDIR", "RANDOM_STATE", "VAE_EPOCHS", "MLP_EPOCHS"}
    missing = required - values.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Configuration file {config_path} is missing required settings: {names}")

    def resolve_path(key: str) -> Path:
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration setting {key} must be a non-empty path string")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"Configuration setting {key} must be an absolute path")
        return path.resolve()

    def resolve_int(key: str) -> int:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Configuration setting {key} must be an integer")
        return value

    random_state = resolve_int("RANDOM_STATE")
    vae_epochs = resolve_int("VAE_EPOCHS")
    mlp_epochs = resolve_int("MLP_EPOCHS")

    return WorkflowConfig(
        acx=resolve_path("ACX"),
        uedge_campaign_rootdir=resolve_path("UEDGE_CAMPAIGN_ROOTDIR"),
        random_state=random_state,
        vae_epochs=vae_epochs,
        mlp_epochs=mlp_epochs,
    )


def input_yes_or_no(msg: str, default_answer: bool = False) -> bool:
    ret = default_answer
    print(msg, end="")
    while True:
        answer = input().lower()
        if answer in ("n", "no"):
            ret = False
            break
        if answer in ("y", "yes"):
            ret = True
            break
        print("Answer y[es] or n[o]: ", end="")
    return ret


def input_int(prompt: str, min_val: int, max_val: int, default: int) -> int:
    while True:
        user_input = input(f"{prompt} [{min_val}-{max_val}] (default={default}): ").strip()

        # If user just hits Enter → return default
        if user_input == "":
            return default

        # Try converting to integer
        try:
            value = int(user_input)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        # Check bounds
        if value < min_val or value > max_val:
            print(f"Please enter a number between {min_val} and {max_val}.")
            continue

        return value
