# KSTAR UEDGE training_workflow

The `UEDGE Simulation Dataset for KSTAR Divertor Detachment Studies` is published at https://portal.nersc.gov/cfs/m3976/kstar_uedge.html

This repository contains the model training workflow (a python script) that uses the original DivControlNN training code while it uses hpc-campaign files for access to the entire published simulation dataset hosted at NERSC. It can be used to re-create the same model that was published by the DivControlNN team, or can be modified to read more data and prepare a different training dataset.

Original training code is from https://github.com/LLNL-FusionML/DivControlNN.git

## Installation
```bash
git clone https://github.com/pnorbert/kstar_uedge_training_workflow
cd kstar_uedge_training_workflow
python3 -m venv .venv
. .venv/bin/activate
pip3 install -e .[cuda]
```

## Preparation
You need the campaign files to access the data. If you are running on Perlmutter, you have direct access to the data on disk. If you are running elsewhere, you need to set up the public remote access to the data. 

### Preparation for running on Perlmutter
in ./config.yaml set
```bash
ACX: /global/cfs/cdirs/m3976/www/campaign-store/kstar.acx
UEDGE_campaign_rootdir: /global/cfs/cdirs/m3976/www/campaign-store/KSTAR24
```
The random state (seed) is used to replicate the same download order every time when starting from scratch. The workflow can be restarted and download more data and re-train the model on more data, using a fixed random seed. 

### Preparation for running remotely
Familiarize yourself with hpc-campaign at https://hpc-campaign.readthedocs.io

Set up a dedicated folder for storing campaigns, e.g. `~/campaign-store`, or better yet, a folder under your favorite cloud file storage and sharing service's root folder. Set up `~/.config/hpc-campaign/config.yaml` and `~/.config/hpc-campaign/hosts.yaml` as discussed in the hpc-campaign documentation.

Download the campaign files from https://portal.nersc.gov/cfs/m3976/campaign-store-KSTAR24.tar.gz, 
and extract it under your campaign store folder.

in ./config.yaml set
```bash
ACX: <your-campaign-store-absolute-path>/kstar.acx
UEDGE_campaign_rootdir: <your-campaign-store-absolute-path>/KSTAR24
```

Create the `kstar.acx` campaign-index (or download from https://portal.nersc.gov/cfs/m3976/campaign-store/kstar.acx)
```bash
hpc_campaign index kstar.acx add KSTAR24
hpc_campaign ls -x
```

### Common preparation, test

Listing should show KSTAR24 campaigns
```bash
hpc_campaign ls
KSTAR24/Ip800_p3.5_d1.7999999999999998.aca
KSTAR24/Ip800_p3.75_d1.2.aca
...
KSTAR24/Ip700_p1.5428571428571427_d1.5166666666666666.aca
```

Campaign manager info should give you detailed info about one campaign file
```bash
hpc_campaign manager KSTAR24/Ip800_p3.75_d1.2.aca info

```

## Partitions of the dataset
There are 4 partitions:

- training set
- validation set
- test set
- final evaluation set

The DivControlNN package is using the first three partitions to train a model. The final evaluation set is never used during training. Rather, it should be used for evaluating a model later. If you don't need such final evaluation set and want to train on all samples, choose to download 0 samples for it.

## Running the workflow
Copy or edit `config.yaml` so that `ACX` and `UEDGE_campaign_rootdir` contain absolute paths to the campaign data, and pick a `RANDOM_STATE` other than 1. 
then run:

```bash
kstar_uedge_download_final_evaluation_set
kstar_uedge_trainer
```


## Original training and final evaluation sets by Ben Zhu et al.

The original case lists used for the DivControlNN publication can be downloaded in their published order with:

```bash
kstar_uedge_download_original_set --dataset final_evaluation
kstar_uedge_download_original_set --dataset training
```

The final evaluation and training case lists are `original/cases_for_final_evaluation.txt` and
`original/cases_for_training.txt`. For the original publication, the authors set aside `20%` of the dataset for a final evaluation, that was not used in the training process at all. 

Outputs are written to `final_evaluation_set/16293_1` or `training_set/1`.

To recreate the original training, 

- first make a link in the training set to the final evaluation set, 
- in config.yaml, change to `RANDOM_STATE: 1`
- then run the trainer and ask to download `0` more samples.

```bash
cd training_set/1
ln -s ../../final_evaluation_set/16293_1 final_evaluation_set
cd ../../
# in config.yaml: set RANDOM_STATE to 1
kstar_uedge_trainer
```


## DivControlNN publication
```
Zhao, M., Xu, X. Q., Zhu, B., et al.,
“Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR,”
submitted to Nuclear Fusion (2025).
arXiv: https://arxiv.org/abs/2510.16199
```
