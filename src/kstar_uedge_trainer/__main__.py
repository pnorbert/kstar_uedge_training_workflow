"""Command-line entry point for the training workflow."""

import runpy


def main() -> None:
    """Run the trainer module as a script."""
    runpy.run_module("kstar_uedge_trainer.trainer", run_name="__main__")


if __name__ == "__main__":
    main()
