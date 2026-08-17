# kstar_uedge_training_workflow
DivControlNN recreated using hpc-campaign files and remote reading

Original training code is from https://github.com/LLNL-FusionML/DivControlNN.git

Copy or edit `config.yaml` so that `ACX` and `UEDGE_campaign_rootdir` contain absolute paths to the campaign data,
then run:

```bash
kstar_uedge_download_testing_set
kstar_uedge_trainer
```

## Original training and testing sets by Ben Zhu et al.

The original case lists used for the DivControlNN publication can be downloaded in their published order with:

```bash
kstar_uedge_download_original_set --dataset testing
kstar_uedge_download_original_set --dataset training
```

The testing and training case lists are `original/cases_for_testing.txt` and `original/cases_for_training.txt`.
Outputs are written to `testing_set/<n>_1` or `training_set/1`; by default, `n` is the full list length.

## DivControlNN publication
```
Zhao, M., Xu, X. Q., Zhu, B., et al.,
“Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR,”
submitted to Nuclear Fusion (2025).
arXiv: https://arxiv.org/abs/2510.16199
```
