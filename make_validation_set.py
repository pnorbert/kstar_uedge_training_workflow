import sys
from parameters import GetParameters, GetDataFrame
from loader import read_one_campaign, combine_data
from utils import input_int, input_yes_or_no
from pathlib import Path
from shutil import rmtree

#
# MAIN program
#

ACX = Path("/home/adios/dropbox/adios-campaign-store/kstar.acx").resolve()
UEDGE_campaign_rootdir = Path(
    "/home/adios/dropbox/adios-campaign-store/KSTAR24"
).resolve()
RANDOM_STATE = 42
VALIDATION_DIR = Path("validation_set")

if not ACX.exists():
    print(f"ERROR: Campaign index file {ACX} does not exist")
    sys.exit(1)

if not UEDGE_campaign_rootdir.exists():
    print(f"ERROR: Directory {UEDGE_campaign_rootdir} does not exist")
    sys.exit(1)

# 1. Get all the available runs (from ACX) -> DataFrame  (parameters.py )
# 2. Get M random samples from the DataFrame for Validation set. Save to disk (pickle)
# 3. Get the archives (aca) and runs that need to be read
# 4. Read the runs and process the data -> data for validating. Save to disk (adios)

#
# 1. Get all the available runs (from ACX) -> DataFrame
#
Ip_list, p_list, d_list, n_list, f_list = GetParameters(str(ACX))
df = GetDataFrame(Ip_list, p_list, d_list, n_list, f_list)

print(
    """
   Cases = Number of cases to traing on. 
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
m_samples = input_int("Number of validation samples", 10, int(len(df) / 3), 1000)

savedir = VALIDATION_DIR / f"{m_samples}_{RANDOM_STATE}"
if (savedir / "df.pkl").exists() and (savedir / "validation_set.bp").exists():
    print(
        f"This validation set ({m_samples} samples with random state {RANDOM_STATE}) already exist"
    )
    if not input_yes_or_no("Do you want to recreate this sample (y/n)? "):
        sys.exit(0)
    rmtree(savedir / "validation_set.bp")
    (savedir / "df.pkl").unlink()
else:
    savedir.mkdir(parents=True, exist_ok=True)

sampled_df = df.sample(m_samples, random_state=RANDOM_STATE)
sampled_df.to_pickle(savedir / "df.pkl")

#
# 3. Get the archives (aca) and runs that need to be read
#
grouped = sampled_df.groupby(["Ip", "p", "d"])[["n", "f"]].apply(
    lambda g: list(map(tuple, g.to_numpy()))
)
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

ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = combine_data(
    output=savedir / "validation_set.bp", append=False
)

print(f"Shape of rads = {rads.shape}")
# print(rads)
print(
    f"Validation set is saved in {savedir}  with {m_samples} samples using random state {RANDOM_STATE}"
)
