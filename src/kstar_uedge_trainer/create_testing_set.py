import sys
from pathlib import Path
from shutil import rmtree

from .loader import combine_data, read_one_campaign
from .parameters import GetDataFrame, GetParameters
from .utils import input_int, input_yes_or_no, read_config

CONFIG = read_config()
ACX = CONFIG.acx
UEDGE_campaign_rootdir = CONFIG.uedge_campaign_rootdir
RANDOM_STATE = CONFIG.random_state
TESTING_DIR = Path("testing_set")


def main() -> None:
    """Create a testing set from randomly selected campaign runs."""
    if not ACX.exists():
        print(f"ERROR: Campaign index file {ACX} does not exist")
        sys.exit(1)

    if not UEDGE_campaign_rootdir.exists():
        print(f"ERROR: Directory {UEDGE_campaign_rootdir} does not exist")
        sys.exit(1)

    # 1. Get all the available runs (from ACX) -> DataFrame  (parameters.py )
    # 2. Get M random samples from the DataFrame for Validation set. Save to disk (pickle)
    # 3. Get the archives (aca) and runs that need to be read
    # 4. Read the runs and process the data -> data for evaluation. Save to disk (adios)

    #
    # 1. Get all the available runs (from ACX) -> DataFrame
    #
    Ip_list, p_list, d_list, n_list, f_list = GetParameters(str(ACX))
    df = GetDataFrame(Ip_list, p_list, d_list, n_list, f_list)

    print(
        """
   Cases = Number of cases to train on.
   ip    = plasma current [in kA],
   n     = ncore: core electron (roughly psin=0.95) density [in m^-3]
   p     = pinj:  total injection power [in MW]
   f     = fz:    impurity fraction
   d     = diff:  diffusion coefficient scaling factor
"""
    )

    print(f"Cases = {len(df)}")

    #
    # 2. Get M random samples from the DataFrame
    #
    m_samples = input_int("Number of testing samples", 10, int(len(df) / 3), 1000)

    savedir = TESTING_DIR / f"{m_samples}_{RANDOM_STATE}"
    if (savedir / "df.pkl").exists() and (savedir / "testing_set.bp").exists():
        print(f"This testing set ({m_samples} samples with random state {RANDOM_STATE}) already exist")
        if not input_yes_or_no("Do you want to recreate this sample (y/n)? "):
            sys.exit(0)
        rmtree(savedir / "testing_set.bp")
        (savedir / "df.pkl").unlink()
    else:
        savedir.mkdir(parents=True, exist_ok=True)

    sampled_df = df.sample(m_samples, random_state=RANDOM_STATE)
    sampled_df.to_pickle(savedir / "df.pkl")

    #
    # 3. Get the archives (aca) and runs that need to be read
    #
    grouped = sampled_df.groupby(["Ip", "p", "d"])[["n", "f"]].apply(lambda g: list(map(tuple, g.to_numpy())))
    # print(f"grouped: {type(grouped)}")

    #
    # 4. Read the runs and process the data -> data for training
    #
    cases_count = 0
    for (Ip, p, d), nf_pairs in grouped.items():
        ACA = UEDGE_campaign_rootdir / f"Ip{Ip}_p{p}_d{d}.aca"
        print(f"    Ip{Ip}_p{p}_d{d}.aca:")
        cases_count += read_one_campaign(ACA, Ip, p, d, nf_pairs)
        print(f"      {cases_count} cases downloaded.")
    print(f"In total, {cases_count} cases are attained.")

    *_, rads = combine_data(output=savedir / "testing_set.bp", append=False)

    print(f"Shape of rads = {rads.shape}")
    # print(rads)
    print(f"Validation set is saved in {savedir}  with {m_samples} samples using random state {RANDOM_STATE}")


if __name__ == "__main__":
    main()
