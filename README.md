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
in `./workflow_config.yaml` set
```bash
ACX: /global/cfs/cdirs/m3976/www/campaign-store/kstar.acx
UEDGE_CAMPAIGN_ROOTDIR: /global/cfs/cdirs/m3976/www/campaign-store/KSTAR24

RANDOM_STATE: 42
VAE_EPOCHS: 5000
MLP_EPOCHS: 3000
```
Other parameters: 

- `RANDOM_STATE`: (seed) is used to replicate the same download order every time when starting from scratch. The workflow can be restarted and download more data and re-train the model on more data, using a fixed random seed. 

- `VAE_EPOCHS` / `MLP_EPOCHS`: the number of epochs to train the VAE and the MLP models. The original paper used 5000 / 3000 epochs to train them that took many hours on an NVidia A100 GPU. 

### Preparation for running remotely
Familiarize yourself with hpc-campaign at https://hpc-campaign.readthedocs.io

Set up a dedicated folder for storing campaigns, e.g. `~/campaign-store`. Set up `~/.config/hpc-campaign/config.yaml` as discussed in the hpc-campaign documentation. The config should point to the campaign store path, under which the KSTAR24/ campaigns will be extracted:

- ~/.config/hpc-campaign/config.yaml

```
Campaign:
  hostname: <whatever_hostname_you_want>
  campaignstorepath: ~/campaign-store
  cachepath: ~/campaign-cache
  verbose: 0
```

Download the campaign files from https://portal.nersc.gov/cfs/m3976/campaign-store-KSTAR24.tar.gz, 
and extract it under your campaign store folder.

Create the `kstar.acx` campaign-index (or download from https://portal.nersc.gov/cfs/m3976/campaign-store/kstar.acx)
```bash
hpc_campaign index kstar.acx add KSTAR24
hpc_campaign ls -x
```

in `./workflow_config.yaml` set
```bash
ACX: <your-campaign-store-absolute-path>/kstar.acx
UEDGE_CAMPAIGN_ROOTDIR: <your-campaign-store-absolute-path>/KSTAR24
```


### Common preparation, test

Listing should show KSTAR24 campaigns. If you did not set up ~/.config/hpc-campaign/config.yaml, you need to point to the path where the campaign files are located manually:

```bash
hpc_campaign ls [--campaign_store <your-campaign-store-absolute-path>] KSTAR24
KSTAR24/Ip800_p3.5_d1.7999999999999998.aca
KSTAR24/Ip800_p3.75_d1.2.aca
...
KSTAR24/Ip700_p1.5428571428571427_d1.5166666666666666.aca
```

Campaign manager info should give you detailed info about one campaign file
```bash
hpc_campaign manager [--campaign_store <your-campaign-store-absolute-path>] KSTAR24/Ip800_p3.75_d1.2.aca info

```

## Partitions of the dataset
There are 4 partitions:

- training set
- validation set
- test set
- final evaluation set

The DivControlNN package is using the first three partitions to train a model. The final evaluation set is never used during training. Rather, it should be used for evaluating a model later. If you don't need such final evaluation set and want to train on all samples, choose to download 0 samples for it.

## Running the workflow
Copy or edit `workflow_config.yaml` so that `ACX` and `UEDGE_CAMPAIGN_ROOTDIR` contain absolute paths to the campaign data, and pick a `RANDOM_STATE` other than 1. 
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
`original/cases_for_training.txt`. For the original publication, the authors set aside `20%` of the dataset for a final evaluation, that was not used in the training process at all. The rest of the samples (`80%`) is split into `72% / 18% / 10%` training/validation/test cases by the DivControlNN package.

Outputs are written to `final_evaluation_set/16293_1` or `training_set/1`.

To recreate the original training, 

- first make a link in the training set to the final evaluation set, 
- in workflow_config.yaml, change to `RANDOM_STATE: 1`
- use `VAE_EPOCHS: 5000` and `MLP_EPOCHS: 3000` to train the models for long. 
- then run the trainer and ask to download `0` more samples.

```bash
cd training_set/1
ln -s ../../final_evaluation_set/16293_1 final_evaluation_set
cd ../../
# in workflow_config.yaml: set RANDOM_STATE to 1
kstar_uedge_trainer
```


## DivControlNN publication
```
Ben Zhu, Menglong Zhao, Xue-Qiao Xu, Anchal Gupta, KyuBeen Kwon, Xinxing Ma, David Eldon
Latent space mapping: Revolutionizing predictive models for divertor plasma detachment control.
Phys. Plasmas 1 June 2025; 32 (6): 062508.
```
https://doi.org/10.1063/5.0267930

