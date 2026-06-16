import pandas as pd
import sys
from parameters import (
    GetParameters,
    GetDataFrame,
    select_validation_dir,
    ask_an_integer,
)
from loader import read_one_campaign, combine_data, load_data
from utils import input_int, input_yes_or_no
from pathlib import Path
from DivControlNN.train_autoencoder import train_autoencoder

#
# MAIN program
#

ACX = Path("/home/adios/dropbox/adios-campaign-store/kstar.acx").resolve()
UEDGE_campaign_rootdir = Path(
    "/home/adios/dropbox/adios-campaign-store/KSTAR24"
).resolve()
RANDOM_STATE = 42
VALIDATION_DIR = Path("validation_set")
TRAININGSET_DIR = Path(f"training_set/{RANDOM_STATE}")
MODEL_DIR = Path(f"model")

if not ACX.exists():
    print(f"ERROR: Campaign index file {ACX} does not exist")
    sys.exit(1)

if not UEDGE_campaign_rootdir.exists():
    print(f"ERROR: Directory {UEDGE_campaign_rootdir} does not exist")
    sys.exit(1)

# 1. Check existing training data that is already downloaded
# 2. Select validation set first, load validation parameters (dataframe)
# 3. Get all the available runs (from ACX) -> DataFrame  (parameters.py )
# in a loop until model is good enough
#   4. Get N random samples from the Training set
#   5. Get the archives (aca) and runs that need to be read
#   6. Read the runs and process the data -> data for training
#   7. Train model
#   8. Validate model
#   9. Select new N random samples

#
# 1. Check on existing training data
#
validation_set_dir = None
df_existing_training = pd.DataFrame()
if (TRAININGSET_DIR / "df.pkl").exists() and (
    TRAININGSET_DIR / "training_set.bp"
).exists():
    print(f"Some training set exists in {TRAININGSET_DIR}")
    df_existing_training: pd.DataFrame = pd.read_pickle(TRAININGSET_DIR / "df.pkl")
    print(f"    found {len(df_existing_training)} samples")
    tsd = TRAININGSET_DIR / "validation_set"
    if tsd.exists() and (tsd.is_symlink() or tsd.is_dir()):
        validation_set_dir = tsd

#
# 2. Select validation set first, load validation parameters (dataframe)
#
if validation_set_dir is None:
    validation_set_dir = select_validation_dir(VALIDATION_DIR)
    if validation_set_dir is None:
        print(f"Run make_validation_set.py to create a validation set first")
        sys.exit(1)

    if (
        not (validation_set_dir / "df.pkl").exists()
        or not (validation_set_dir / "validation_set.bp").exists()
    ):
        print(
            f"The validation set in {validation_set_dir} is missing/incomplete. Rerun make_validation_set.py"
        )
        sys.exit(1)

df_validation = pd.read_pickle(validation_set_dir / "df.pkl")


#
# 3. Get all the available runs (from ACX) -> DataFrame
#
Ip_list, p_list, d_list, n_list, f_list = GetParameters(str(ACX))
df = GetDataFrame(Ip_list, p_list, d_list, n_list, f_list)
df_all_training = df.drop(df_validation.index)
if not df_existing_training.empty:
    df_all_training = df_all_training.drop(df_existing_training.index)

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

print(f"Cases = {len(df_all_training)}")

# for # max batches
# choose random N points
# preprocess data
# train
# test accuracy

# 4. Get N random samples from the DataFrame
#
n_samples = ask_an_integer("How many samples to read (0:exit)? ")
if n_samples < 1:
    sys.exit(1)
sampled_df = df_all_training.sample(n_samples, random_state=RANDOM_STATE)

#
# 5. Get the archives (aca) and runs that need to be read
#
grouped = sampled_df.groupby(["Ip", "p", "d"])[["n", "f"]].apply(
    lambda g: list(map(tuple, g.to_numpy()))
)

#
# 6. Read the runs and process the data -> data for training
#
cases_count = 0
for (Ip, p, d), nf_pairs in grouped.items():
    ACA = UEDGE_campaign_rootdir / f"Ip{Ip}_p{p}_d{d}.aca"
    print(f"    Ip{Ip}_p{p}_d{d}.aca:")
    cases_count += read_one_campaign(ACA, Ip, p, d, nf_pairs)
    print(f"      {cases_count} cases downloaded.")
print(f"In total, {cases_count} cases were downloaded.")


if (TRAININGSET_DIR / "df.pkl").exists() and (
    TRAININGSET_DIR / "training_set.bp"
).exists():
    print(f"Append new data to training set {TRAININGSET_DIR}")
    df_existing_training: pd.DataFrame = pd.read_pickle(TRAININGSET_DIR / "df.pkl")
    merged_df = pd.concat(
        [df_existing_training, sampled_df], ignore_index=True
    ).drop_duplicates()
    merged_df.to_pickle(TRAININGSET_DIR / "df.pkl")
else:
    print(f"Save data to training set {TRAININGSET_DIR}")
    TRAININGSET_DIR.mkdir(parents=True, exist_ok=True)
    sampled_df.to_pickle(TRAININGSET_DIR / "df.pkl")
    (TRAININGSET_DIR / "validation_set").symlink_to(
        validation_set_dir.resolve(), target_is_directory=True
    )


# we could keep data in memory and update the model but instead
# here we save the new data as another step in the training set on disk
# and reload the whole thing for trainign from file
# ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = combine_data(output=savedir)
# print(f"Shape of rads = {rads.shape}")
# print(rads)
combine_data(output=TRAININGSET_DIR / "training_set.bp", append=True)

#
#   7. Train model
#

ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = load_data(
    TRAININGSET_DIR / "training_set.bp"
)

n = len(df_existing_training) + n_samples
train_autoencoder(TRAININGSET_DIR, f"{RANDOM_STATE}_{n}")


#   8. Validate model
#   9. Select new N random samples
