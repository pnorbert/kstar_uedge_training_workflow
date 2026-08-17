"""Download an original testing or training set in its published case order."""

import argparse
import json
import re
import sys
from importlib.metadata import distributions
from pathlib import Path
from shutil import rmtree
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
from adios2 import Adios, FileReader, Stream

import kstar_uedge_trainer

from .parameters import GetDataFrame, GetParametersFromCaseList
from .utils import input_yes_or_no, read_config

CONFIG = read_config()
UEDGE_CAMPAIGN_ROOTDIR = CONFIG.uedge_campaign_rootdir
CASE_LIST_FILENAMES = {
    "testing": "cases_for_testing.txt",
    "training": "cases_for_training.txt",
}
OUTPUT_ROOTS = {
    "testing": Path("testing_set"),
    "training": Path("training_set"),
}
ELEMENTARY_CHARGE = 1.60217663e-19


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")


def _find_case_list(dataset: str) -> Path:
    """Locate a case list from an editable source tree, then pip metadata."""
    filename = CASE_LIST_FILENAMES[dataset]

    package_file = kstar_uedge_trainer.__file__
    if package_file is not None:
        source_case_list = (Path(package_file).resolve().parent / ".." / ".." / "original" / filename).resolve()
        if source_case_list.is_file():
            return source_case_list

    source_urls = []
    for package in distributions(name="kstar-uedge-trainer"):
        direct_url_text = package.read_text("direct_url.json")
        if direct_url_text is None:
            continue
        try:
            source_url = json.loads(direct_url_text)["url"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        source_urls.append(source_url)

        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "file":
            continue
        repository = Path(url2pathname(unquote(parsed_url.path))).resolve()
        if not (repository / ".git").exists():
            continue

        case_list = repository / "original" / filename
        if not case_list.is_file():
            raise FileNotFoundError(f"Case list does not exist: {case_list}")
        return case_list

    if not source_urls:
        raise FileNotFoundError("The pip installation has no direct_url.json with a source checkout location")
    raise FileNotFoundError(f"No pip source location is an existing local Git repository: {source_urls}")


def _remove_output(path: Path) -> None:
    if path.is_dir():
        rmtree(path)
    elif path.exists():
        path.unlink()


def _output_directory(dataset: str, number: int) -> Path:
    name = f"{number}_1" if dataset == "testing" else "1"
    return OUTPUT_ROOTS[dataset] / name


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = _ArgumentParser(
        description="Download an original testing or training set in case-list order.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(CASE_LIST_FILENAMES),
        required=True,
        help="case list to download",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of cases to download from the start of the list (default: the entire list)",
    )
    args = parser.parse_args(argv)
    if args.number is not None and args.number < 1:
        parser.error("--number must be a positive integer")
    return args


def _campaign_and_image_path(row: Any) -> tuple[Path, str]:
    campaign = UEDGE_CAMPAIGN_ROOTDIR / f"Ip{row.Ip}_p{row.p}_d{row.d}.aca"
    image_path = f"n{row.n}/f{row.f}/images"
    return campaign, image_path


def _included_datasets(df: pd.DataFrame) -> dict[Path, list[str]]:
    """Collect dataset filters without changing the dataframe's row order."""
    datasets: dict[Path, list[str]] = {}
    for row in df.itertuples(index=False):
        campaign, image_path = _campaign_and_image_path(row)
        datasets.setdefault(campaign, []).append(image_path)
    return datasets


def _read_diagnostics(
    reader: FileReader,
    variables: set[str],
    campaign: Path,
    image_path: str,
) -> tuple[np.ndarray, ...]:
    required = (
        "ni",
        "te",
        "qtr_new",
        "qtl_new",
        "qradhl",
        "qradzl",
        "rads",
    )
    missing = [name for name in required if f"{image_path}/{name}" not in variables]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing variables ({names}) in {campaign} / {image_path}")

    if all(f"{image_path}/{name}" in variables for name in ("jsatr", "jsatl")):
        jr_name, jl_name = "jsatr", "jsatl"
    elif all(f"{image_path}/{name}" in variables for name in ("jr", "jl")):
        print(f"    WARNING: No jsatl/jsatr, using jr/jl from {campaign.name} / {image_path}")
        jr_name, jl_name = "jr", "jl"
    else:
        raise RuntimeError(f"Missing ion saturation current in {campaign} / {image_path}")

    ni_variable = reader.inquire_variable(f"{image_path}/ni")
    if ni_variable is None:
        raise RuntimeError(f"Cannot inquire ni in {campaign} / {image_path}")
    ni_variable.set_selection([[33, 1, 0], [1, ni_variable.shape()[1] - 2, 1]])

    jr = reader.read(f"{image_path}/{jr_name}", defer_read=True)
    jl = reader.read(f"{image_path}/{jl_name}", defer_read=True)
    neu = reader.read(ni_variable, defer_read=True)
    te = reader.read(f"{image_path}/te", defer_read=True)
    qtr_new = reader.read(f"{image_path}/qtr_new", defer_read=True)
    qtl_new = reader.read(f"{image_path}/qtl_new", defer_read=True)
    qradhl = reader.read(f"{image_path}/qradhl", defer_read=True)
    qradzl = reader.read(f"{image_path}/qradzl", defer_read=True)
    rads = reader.read(f"{image_path}/rads", defer_read=True)
    reader.read_complete()

    qtl = qtl_new[:] - 2.0 * (qradhl[:] + qradzl[:])
    return (
        np.squeeze(neu) / 1e19,
        te[33, 1:-1] / ELEMENTARY_CHARGE,
        te[-1, 1:-1] / ELEMENTARY_CHARGE,
        te[1, 1:-1] / ELEMENTARY_CHARGE,
        jr[1:-1],
        -jl[1:-1],
        qtr_new[1:-1],
        -qtl[1:-1],
        rads[:],
    )


def _download_in_dataframe_order(df: pd.DataFrame) -> dict[str, np.ndarray]:
    included_datasets = _included_datasets(df)
    readers: dict[Path, FileReader] = {}
    variables: dict[Path, set[str]] = {}
    adios_contexts: dict[Path, Adios] = {}

    values: dict[str, list] = {
        name: []
        for name in (
            "ip",
            "ncore",
            "pinj",
            "fz",
            "diff",
            "neu",
            "teu",
            "ter",
            "tel",
            "jr",
            "jl",
            "qtr",
            "qtl",
            "rads",
        )
    }

    try:
        for case_number, row in enumerate(df.itertuples(index=False), start=1):
            campaign, image_path = _campaign_and_image_path(row)
            if campaign not in readers:
                print(f"    Opening {campaign.name}")
                adios_contexts[campaign] = Adios()
                io = adios_contexts[campaign].declare_io("campaign-reader")
                filters = ";".join(re.escape(path) for path in included_datasets[campaign])
                io.set_parameter("include-dataset", filters)
                readers[campaign] = FileReader(io, str(campaign))
                variables[campaign] = set(readers[campaign].available_variables())

            diagnostics = _read_diagnostics(readers[campaign], variables[campaign], campaign, image_path)
            values["ip"].append(float(row.Ip))
            values["ncore"].append(float(row.n))
            values["pinj"].append(2.0 * float(row.p))
            values["fz"].append(float(row.f))
            values["diff"].append(float(row.d))
            for name, value in zip(
                ("neu", "teu", "ter", "tel", "jr", "jl", "qtr", "qtl", "rads"),
                diagnostics,
                strict=True,
            ):
                values[name].append(value)

            if case_number % 100 == 0 or case_number == len(df):
                print(f"      {case_number} of {len(df)} cases downloaded.")
    finally:
        for campaign, reader in readers.items():
            try:
                reader.close()
            except Exception as exc:  # Continue closing every campaign reader.
                print(f"WARNING: Could not close {campaign}: {exc}", file=sys.stderr)

    return {name: np.asarray(value) for name, value in values.items()}


def _verify_parameter_order(df: pd.DataFrame, data: dict[str, np.ndarray]) -> None:
    expected = {
        "ip": df["Ip"].to_numpy(dtype=float),
        "ncore": df["n"].to_numpy(dtype=float),
        "pinj": 2.0 * df["p"].to_numpy(dtype=float),
        "fz": df["f"].to_numpy(dtype=float),
        "diff": df["d"].to_numpy(dtype=float),
    }
    for name, expected_values in expected.items():
        if not np.array_equal(data[name], expected_values):
            raise RuntimeError(f"Downloaded {name} values do not match dataframe row order")


def _write_dataset(output: Path, data: dict[str, np.ndarray]) -> None:
    with Stream(str(output), "w") as stream:
        for name, values in data.items():
            stream.write(name, values)
        stream.write("nsamples", data["ip"].size)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        case_list = _find_case_list(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if not UEDGE_CAMPAIGN_ROOTDIR.is_dir():
        print(f"ERROR: Campaign directory {UEDGE_CAMPAIGN_ROOTDIR} does not exist")
        sys.exit(1)

    df = GetDataFrame(*GetParametersFromCaseList(case_list))
    number = len(df) if args.number is None else args.number
    if number > len(df):
        print(f"ERROR: Requested {number} cases, but {case_list} contains only {len(df)}")
        sys.exit(1)
    df = df.head(number).copy()

    savedir = _output_directory(args.dataset, number)
    dataframe_path = savedir / "df.pkl"
    dataset_path = savedir / f"{args.dataset}_set.bp"

    print("Options:")
    print(f"    dataset:  {args.dataset}")
    print(f"    case list: {case_list}")
    print(f"    number:    {number}")
    print(f"    output:    {savedir}")

    if dataframe_path.exists() or dataset_path.exists():
        print(f"The {args.dataset} set in {savedir} already exists or is incomplete.")
        if not input_yes_or_no("Do you want to recreate it (y/n)? "):
            sys.exit(0)
        _remove_output(dataframe_path)
        _remove_output(dataset_path)

    print(f"Downloading {number} {args.dataset} cases from {case_list} in file order.")
    data = _download_in_dataframe_order(df)
    _verify_parameter_order(df, data)

    savedir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(dataframe_path)
    _write_dataset(dataset_path, data)
    print(f"The {args.dataset} set was saved in {savedir} in the original case-list order.")


if __name__ == "__main__":
    main()
