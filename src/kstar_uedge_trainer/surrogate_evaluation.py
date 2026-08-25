#!/usr/bin/env python3
"""Report prediction-error statistics for one trained KSTAR UEDGE surrogate.

The trainer writes truth and predictions to ``combined_validation.npz`` in the
MLP model directory.  This command evaluates those saved predictions without
loading TensorFlow or importing DivControlNN.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np


class Quantity(NamedTuple):
    name: str
    truth_key: str
    prediction_key: str
    kind: str
    cut: float | None
    unit: str


# These conventions match DivControlNN/surrogate_evaluation.py and Appendix B
# of PoP 32, 062508. Peak-radiation locations use absolute error; all other
# quantities use relative error with the original gross-outlier cuts.
SCALARS = (
    Quantity("f_rad", "rads", "y_rads", "rel", 1.0, ""),
    Quantity("f_rad,div", "rads", "y_rads", "rel", 1.0, ""),
    Quantity("P_rad,max", "rads", "y_rads", "rel", 1.0, ""),
    Quantity("L_peak", "rads", "y_rads", "abs", None, "m"),
    Quantity("R_peak", "rads", "y_rads", "abs", None, "m"),
)
PROFILES = (
    Quantity("Te,omp", "teu", "y_teu", "rel", 0.5, ""),
    Quantity("ne,omp", "neu", "y_neu", "rel", 0.5, ""),
    Quantity("Q_idt", "qtl", "y_qtl", "rel", 0.5, ""),
    Quantity("Jsat_idt", "jl", "y_jl", "rel", 0.5, ""),
    Quantity("Te_idt", "tel", "y_tel", "rel", 0.5, ""),
    Quantity("Q_odt", "qtr", "y_qtr", "rel", 0.5, ""),
    Quantity("Jsat_odt", "jr", "y_jr", "rel", 0.5, ""),
    Quantity("Te_odt", "ter", "y_ter", "rel", 0.5, ""),
)
QUANTITIES = SCALARS + PROFILES

# Current workflow models use the first name. The second name keeps the
# evaluator useful for older DivControlNN model directories.
WEIGHT_FILENAMES = ("best_val.weights.h5", "best_val.weights.hdf5")
VALIDATION_FILENAME = "combined_validation.npz"


class EvaluationError(ValueError):
    """Raised when a model directory or its validation output is invalid."""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved final-evaluation predictions for one MLP surrogate model.",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="MLP model directory, for example models/mlp_dc_n32_l4_42_805",
    )
    parser.add_argument("--csv", type=Path, metavar="FILE", help="also write machine-readable results")
    return parser.parse_args(argv)


def _find_weights(model_dir: Path) -> Path:
    for filename in WEIGHT_FILENAMES:
        path = model_dir / filename
        if path.is_file():
            return path
    expected = " or ".join(WEIGHT_FILENAMES)
    raise EvaluationError(f"No MLP weights found in {model_dir}; expected {expected}")


def _load_validation(path: Path) -> dict[str, np.ndarray]:
    required = {key for quantity in QUANTITIES for key in (quantity.truth_key, quantity.prediction_key)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise EvaluationError(f"{path} is missing arrays: {', '.join(missing)}")
            arrays = {key: np.asarray(archive[key]) for key in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(f"Cannot read validation results {path}: {exc}") from exc

    for truth_key, prediction_key in {(q.truth_key, q.prediction_key) for q in QUANTITIES}:
        truth = arrays[truth_key]
        prediction = arrays[prediction_key]
        if truth.shape != prediction.shape:
            raise EvaluationError(
                f"Shape mismatch in {path}: {truth_key}{truth.shape} != {prediction_key}{prediction.shape}"
            )
        if truth.ndim != 2:
            raise EvaluationError(f"Array {truth_key} in {path} must be two-dimensional, found {truth.shape}")

    if arrays["rads"].shape[1] != len(SCALARS):
        raise EvaluationError(f"Array rads in {path} must have {len(SCALARS)} columns, found {arrays['rads'].shape[1]}")

    profile_shapes = {arrays[q.truth_key].shape for q in PROFILES}
    if len(profile_shapes) != 1:
        raise EvaluationError(f"Profile arrays in {path} do not have a common shape: {sorted(profile_shapes)}")
    if arrays["rads"].shape[0] != next(iter(profile_shapes))[0]:
        raise EvaluationError(f"Scalar and profile sample counts differ in {path}")
    return arrays


def _statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    kind: str,
    cut: float | None,
) -> tuple[float, float, float]:
    error = (prediction - truth).ravel()
    if kind == "rel":
        with np.errstate(divide="ignore", invalid="ignore"):
            error = error / truth.ravel()

    total = error.size
    error = error[np.isfinite(error)]
    if cut is not None:
        error = error[np.abs(error) <= cut]
    if not error.size:
        return float("nan"), float("nan"), 1.0

    # scipy.stats.norm.fit computes these same maximum-likelihood estimates.
    return float(np.mean(error)), float(np.std(error)), 1.0 - error.size / total


def evaluate(arrays: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float, str]]:
    results = {}
    for index, quantity in enumerate(SCALARS):
        results[quantity.name] = _statistics(
            arrays[quantity.prediction_key][:, index],
            arrays[quantity.truth_key][:, index],
            quantity.kind,
            quantity.cut,
        ) + (quantity.unit,)
    for quantity in PROFILES:
        results[quantity.name] = _statistics(
            arrays[quantity.prediction_key],
            arrays[quantity.truth_key],
            quantity.kind,
            quantity.cut,
        ) + (quantity.unit,)
    return results


def _print_group(title: str, names: Sequence[str], results: dict[str, tuple[float, float, float, str]]) -> None:
    print(f"\n=== {title} ===")
    print("  " + f"{'quantity':13s}{'mean':>12s}{'std':>12s}{'trimmed':>12s}")
    for name in names:
        mean, std, trimmed, unit = results[name]
        if not np.isfinite(mean) or not np.isfinite(std):
            print(f"  {name:13s}{'n/a':>12s}{'n/a':>12s}{trimmed * 100:11.2f}%")
            continue
        if unit == "m":
            print(f"  {name:13s}{mean * 100:+11.2f} {std * 100:11.2f} {trimmed * 100:11.2f}%  cm")
        else:
            print(f"  {name:13s}{mean * 100:+11.2f}%{std * 100:11.2f}%{trimmed * 100:11.2f}%")


def _write_csv(path: Path, model_dir: Path, results: dict[str, tuple[float, float, float, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "quantity", "mean", "std", "trimmed_fraction", "unit"])
        for quantity in QUANTITIES:
            mean, std, trimmed, unit = results[quantity.name]
            writer.writerow(
                [model_dir.name, quantity.name, f"{mean:.6g}", f"{std:.6g}", f"{trimmed:.6g}", unit or "relative"]
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    try:
        if not model_dir.is_dir():
            raise EvaluationError(f"Model directory does not exist: {model_dir}")
        weights_path = _find_weights(model_dir)
        validation_path = model_dir / VALIDATION_FILENAME
        if not validation_path.is_file():
            raise EvaluationError(
                f"Validation results do not exist: {validation_path}\n"
                "Run combined-model validation for this model before evaluating it."
            )
        arrays = _load_validation(validation_path)
        results = evaluate(arrays)
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"model:      {model_dir.name}")
    print(f"weights:    {weights_path.name}")
    print(f"validation: {validation_path}")
    profile_points = arrays[PROFILES[0].truth_key].shape[1]
    print(f"samples:    {arrays['rads'].shape[0]} ({profile_points} points)")

    _print_group("Radiation information", [q.name for q in SCALARS], results)
    _print_group(
        "Divertor targets",
        ["Q_idt", "Jsat_idt", "Te_idt", "Q_odt", "Jsat_odt", "Te_odt"],
        results,
    )
    _print_group("Outboard midplane", ["Te,omp", "ne,omp"], results)
    print("\nRelative errors are in percent; peak-location errors are in cm.")
    print("Trimmed is the percentage discarded as non-finite or outside the original error cut.")

    if args.csv is not None:
        csv_path = args.csv.expanduser()
        try:
            _write_csv(csv_path, model_dir, results)
        except OSError as exc:
            print(f"ERROR: Cannot write CSV {csv_path}: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
