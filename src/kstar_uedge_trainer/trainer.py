import sys
from pathlib import Path

import pandas as pd

from DivControlNN.train_autoencoder import train_autoencoder
from DivControlNN.train_mlp import train_mlp

from .loader import combine_data, load_data, read_one_campaign
from .parameters import (
    GetDataFrame,
    GetParameters,
    select_final_evaluation_dir,
)
from .utils import input_int, input_yes_or_no, read_config



def exclude_cases(source: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    CASE_COLUMNS = ["Ip", "p", "d", "n", "f"]
    if excluded.empty:
        return source

    source_keys = pd.MultiIndex.from_frame(source[CASE_COLUMNS])
    excluded_keys = pd.MultiIndex.from_frame(excluded[CASE_COLUMNS])
    return source.loc[~source_keys.isin(excluded_keys)]


#
# MAIN program
#

CONFIG = read_config()
ACX = CONFIG.acx
UEDGE_campaign_rootdir = CONFIG.uedge_campaign_rootdir
RANDOM_STATE = CONFIG.random_state
FINAL_EVALUATION_DIR = Path("final_evaluation_set")
TRAININGSET_DIR = Path(f"training_set/{RANDOM_STATE}")
MODEL_DIR = Path("model")

if not ACX.exists():
    print(f"ERROR: Campaign index file {ACX} does not exist")
    sys.exit(1)

if not UEDGE_campaign_rootdir.exists():
    print(f"ERROR: Directory {UEDGE_campaign_rootdir} does not exist")
    sys.exit(1)

# 1. Check existing training data that is already downloaded
# 2. Select final_evaluation set first, load final_evaluation parameters (dataframe)
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
final_evaluation_set_dir = None
df_existing_training = pd.DataFrame()
if (TRAININGSET_DIR / "df.pkl").exists() and (TRAININGSET_DIR / "training_set.bp").exists():
    print(f"Some training set exists in {TRAININGSET_DIR}")
    df_existing_training: pd.DataFrame = pd.read_pickle(TRAININGSET_DIR / "df.pkl")
    print(f"    found {len(df_existing_training)} samples")
    fesd = TRAININGSET_DIR / "final_evaluation_set"
    if fesd.exists() and (fesd.is_symlink() or fesd.is_dir()):
        final_evaluation_set_dir = fesd

#
# 2. Select final_evaluation set first, load final_evaluation parameters (dataframe)
#
if final_evaluation_set_dir is None:
    final_evaluation_set_dir = select_final_evaluation_dir(FINAL_EVALUATION_DIR)
    if final_evaluation_set_dir is None:
        print("Run kstar_uedge_download_final_evaluation_set to download a final evaluation set first")
        sys.exit(1)

    if (
        not (final_evaluation_set_dir / "df.pkl").exists()
        or not (final_evaluation_set_dir / "final_evaluation_set.bp").exists()
    ):
        print(
            f"The final evaluation set in {final_evaluation_set_dir} is missing/incomplete. "
            "Rerun kstar_uedge_download_final_evaluation_set"
        )
        sys.exit(1)

df_final_evaluation = pd.read_pickle(final_evaluation_set_dir / "df.pkl")


#
# 3. Get all the available runs (from ACX) -> DataFrame
#
Ip_list, p_list, d_list, n_list, f_list = GetParameters(str(ACX))
df = GetDataFrame(Ip_list, p_list, d_list, n_list, f_list)
df_all_training = exclude_cases(df, df_final_evaluation)
df_all_training = exclude_cases(df_all_training, df_existing_training)
    
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

print(f"Cases = {len(df_all_training)}")

# for # max batches
# choose random N points
# preprocess data
# train
# test accuracy

# 4. Get N random samples from the DataFrame
#
n_samples = input_int("How many new samples to read (-1:exit 0:train on existing samples)? ", -1, len(df_all_training), 1000)
if n_samples < 0:
    sys.exit(1)

if n_samples == 0 and df_existing_training.empty:
    print(f"No existing samples and no new samples. Quit.")
    sys.exit(1)

if n_samples > 0:
    sampled_df = df_all_training.sample(n_samples, random_state=RANDOM_STATE)

    #
    # 5. Get the archives (aca) and runs that need to be read
    #
    grouped = sampled_df.groupby(["Ip", "p", "d"])[["n", "f"]].apply(lambda g: list(map(tuple, g.to_numpy())))

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

    # we could keep data in memory and update the model but instead
    # here we save the new data as another step in the training set on disk
    # and reload the whole thing for trainign from file
    # ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = combine_data(output=savedir)
    # print(f"Shape of rads = {rads.shape}")
    # print(rads)
    combine_data(output=TRAININGSET_DIR / "training_set.bp", append=True)

# Save all samples into pickle
if not df_existing_training.empty:
    if n_samples > 0:
        print(f"Append new data to training set {TRAININGSET_DIR}")
        merged_df = pd.concat([df_existing_training, sampled_df], ignore_index=False).drop_duplicates()
        merged_df.to_pickle(TRAININGSET_DIR / "df.pkl")
    else:
        merged_df = df_existing_training
else:
    print(f"Save data to training set {TRAININGSET_DIR}")
    TRAININGSET_DIR.mkdir(parents=True, exist_ok=True)
    sampled_df.to_pickle(TRAININGSET_DIR / "df.pkl")
    (TRAININGSET_DIR / "final_evaluation_set").symlink_to(final_evaluation_set_dir.resolve(), target_is_directory=True)

#
#   7. Train model
#

ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = load_data(TRAININGSET_DIR / "training_set.bp")

n = len(df_existing_training) + n_samples
model_id = f"{RANDOM_STATE}_{n}"
autoencoder_path = train_autoencoder(TRAININGSET_DIR, model_id)
train_mlp(TRAININGSET_DIR, autoencoder_path, model_id)


#   9. Select new N random samples
