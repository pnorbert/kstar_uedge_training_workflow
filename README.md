# kstar_uedge_training_workflow
DivControlNN recreated using hpc-campaign files and remote reading

Original training code is from https://github.com/LLNL-FusionML/DivControlNN.git

Copy or edit `config.yaml` so that `ACX` and `UEDGE_campaign_rootdir` contain absolute paths to the campaign data,
then run:

```bash
kstar_uedge_create_testing_set
kstar_uedge_trainer
```
