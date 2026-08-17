# kstar_uedge_training_workflow
DivControlNN recreated using hpc-campaign files and remote reading

Original training code is from https://github.com/LLNL-FusionML/DivControlNN.git

Copy or edit `config.yaml` so that `ACX` and `UEDGE_campaign_rootdir` contain absolute paths to the campaign data,
then run:

```bash
kstar_uedge_download_testing_set
kstar_uedge_trainer
```

## Original final testing set by Ben Zhu et al. 
The original testing set that was used for the DivControlNN publication*, is already included in `original/testing_set/16293_1` but can be re-downloaded with a separate script, which downloads these cases from `original/cases_for_validation.txt`, 

```bash
kstar_uedge_download_original_testing_set
```

This writes the dataframe and ADIOS2 dataset to `original/testing_set/16293_1`.

## DivControlNN publication
```
Zhao, M., Xu, X. Q., Zhu, B., et al.,
“Physics insights from a large-scale 2D UEDGE simulation database for detachment control in KSTAR,”
submitted to Nuclear Fusion (2025).
arXiv: https://arxiv.org/abs/2510.16199
```