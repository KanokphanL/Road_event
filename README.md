# Road_R Challenge
This repository contains our code for the second task of the ROAD-R Challenge.

|   | Precision@0.5 | Recall@0.5 | F1-score@0.5|
| Ours | 0.63 | 0.52 | 0.57 |

our results was ensembled from several model including 5 models from basline and 1 model from mmdection.

We followed the baseline from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep) for data pre and post processing.
From the baseone repository trained and generated results from several backbones including: resnet50RCGRU, resnet50I3D, resnet50C2D, resnet50RCN-NL, resnet50Slowfast, and resnet101Slowfast.



For mmdetection, we added several part to make the detector works with multi-label dataset.


The second task requires that the models' predictions are compliant with 
## ensemble tubes:
put ```ensemble_boxes_wbf_twh.py``` in ```/dfs/data/miniconda/envs/your_env_name/lib/pythonx.x/site-packages/ensemble_boxes/``` <br>
put your tube files into ```./tubes``` <br>
generate ensemble tube files in ```./ensemble_tubes```
### run:
```python ensemble_wbf.py tubes(or your tube path) ensemble_tubes(or your ensemble tube path) tube_file_1.pkl,tube_file_2.pkl,...,tube_file_n.pkl (--dump=map_file_path) (--tuning_iou_thr) (--tuning_skip_box_thr) (tuning_conf_type)```
### example:
```python ensemble_wbf.py tubes ensemble_tubes res50C2D30_test,res50I3D30_test,res50RCLSTM30_test,res50RCN28_test,res50SlowFast30_test test --dump=map.pkl --tuning_iou_thr --tuning_conf_type```

