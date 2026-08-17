"""Download the original testing set from its validation case list."""

import argparse
import sys
from pathlib import Path
from shutil import rmtree

from .loader import combine_data, read_one_campaign
from .parameters import GetDataFrame, GetParametersFromCaseList
from .utils import input_yes_or_no, read_config

CONFIG = read_config()
UEDGE_campaign_rootdir = CONFIG.uedge_campaign_rootdir
CASE_LIST = Path("original/cases_for_validation.txt")
SAVEDIR = Path("original/testing_set/16293_1")
EXPECTED_CASES = 16_293


def _remove_output(path: Path) -> None:
    if path.is_dir():
        rmtree(path)
    elif path.exists():
        path.unlink()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a testing set from the validation case list.")
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="download only the first N cases (for a bounded test run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SAVEDIR,
        help=f"output directory (default: {SAVEDIR})",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.limit is not None and args.output == SAVEDIR:
        parser.error("--output is required with --limit so a partial run cannot overwrite the full testing set")
    return args


def main(argv: list[str] | None = None) -> None:
    """Download the testing set containing every case in the validation case list."""
    args = _parse_args(argv)

    if not CASE_LIST.is_file():
        print(f"ERROR: Validation case list {CASE_LIST} does not exist")
        sys.exit(1)

    if not UEDGE_campaign_rootdir.is_dir():
        print(f"ERROR: Campaign directory {UEDGE_campaign_rootdir} does not exist")
        sys.exit(1)

    Ip_list, p_list, d_list, n_list, f_list = GetParametersFromCaseList(CASE_LIST)
    df = GetDataFrame(Ip_list, p_list, d_list, n_list, f_list)
    if len(df) != EXPECTED_CASES:
        print(f"ERROR: Expected {EXPECTED_CASES} cases in {CASE_LIST}, found {len(df)}")
        sys.exit(1)

    if args.limit is not None:
        if args.limit > len(df):
            print(f"ERROR: --limit {args.limit} exceeds the {len(df)} available cases")
            sys.exit(1)
        df = df.head(args.limit).copy()

    print(
        """
   Cases = Number of cases in the testing set.
   ip    = plasma current [in kA]
   n     = ncore: core electron (roughly psin=0.95) density [in m^-3]
   p     = pinj:  total injection power [in MW]
   f     = fz:    impurity fraction
   d     = diff:  diffusion coefficient scaling factor
"""
    )
    print(f"Cases = {len(df)}")

    savedir: Path = args.output
    dataframe_path = savedir / "df.pkl"
    dataset_path = savedir / "testing_set.bp"
    if dataframe_path.exists() or dataset_path.exists():
        print(f"The testing set in {savedir} already exists or is incomplete.")
        if not input_yes_or_no("Do you want to recreate it (y/n)? "):
            sys.exit(0)
        _remove_output(dataframe_path)
        _remove_output(dataset_path)

    savedir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(dataframe_path)

    grouped = df.groupby(["Ip", "p", "d"])[["n", "f"]].apply(lambda group: list(map(tuple, group.to_numpy())))

    cases_count = 0
    for (Ip, p, d), nf_pairs in grouped.items():
        campaign = UEDGE_campaign_rootdir / f"Ip{Ip}_p{p}_d{d}.aca"
        print(f"    {campaign.name}:")
        cases_count += read_one_campaign(campaign, Ip, p, d, nf_pairs)
        print(f"      {cases_count} cases downloaded.")
    print(f"In total, {cases_count} cases were downloaded.")

    *_, rads = combine_data(output=dataset_path, append=False)

    print(f"Shape of rads = {rads.shape}")
    print(f"Testing set is saved in {savedir} with {len(df)} cases from {CASE_LIST}")
    if cases_count != len(df):
        print(f"WARNING: {len(df) - cases_count} requested cases could not be downloaded")


if __name__ == "__main__":
    main()
