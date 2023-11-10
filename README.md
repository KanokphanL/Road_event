# Road_R Challenge
This repository contains our code for the second task of the ROAD-R Challenge.

|   | Precision@0.5 | Recall@0.5 | F1-score@0.5|
| ------------- | :---: | :---: | :---: |
| Ours | 0.63 | 0.52 | 0.57 |

Our results was ensembled from several model including 5 models from basline and 1 model from mmdection.

## Baseline resutls
We followed the baseline from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep) for data pre and post processing.
From the baseone repository trained and generated results from several backbones including: resnet50RCGRU, resnet50I3D, resnet50C2D, resnet50RCN-NL, resnet50Slowfast, and resnet101Slowfast.

## MMdetection resutls
We applied [mmdetection](https://github.com/open-mmlab/mmdetection) to test multi-label detection.
Our system is based on MMDetection 3.1.0 with some modification to make the detector works with multi-label dataset.

### Training
1. Convert road-r anotation json file to coco format by running ```road_r2coco.py```
2. Train faster-rcnn with agent only for 12 epochs by setting 'with_act' and 'with_loc' to 'False' with faster-rcnn coco pre-trained model
3. Train the agent model with action and location label by setting 'with_act' and 'with_loc' to 'True'
### Testing
1. Generat output.pkl file
2. Convert the output.pkl in mmdetection format to roadr task 1 submission format by running
   ```python mmdet2roadr_out.py mmdet_output.pkl save_dir/ --topk 20 --agent_thres 0.5```

## Ensemble

 - Replace the mmdet folder with the mmdet from the attachment
- Trained faster-rcnn with agent only for 12 epochs by setting 'with_act' and 'with_loc' to 'False' with faster-rcnn coco pre-trained model
- Trained the pre-trained model with 'with_act' and 'with_loc' set to 'True'
- Generated output.pkl file from tools/test.py
- Convert the mmdet output.pkl file to roadr task 1 submission format by running 
	'python mmdet2roadr_out.py mmdet_output.pkl save_dir/ --topk 20 --agent_thres 0.5'

After generating 6 baseline and mmdet output .pkl files, we used ensemble.py to generate the final .pkl file.  
The .pkl path from all models is listed in ensemble_input_list.txt.    	
	'python ensemble.py ensemble_input_list.txt ensemble_road_r_val_pkl/ --topk 20 --skip_box_thr 0.9'

Finally, we generated a submission file for task 2 by running post_processing_raw.py	



The second task requires that the models' predictions are compliant with 
## ensemble tubes:
put ```ensemble_boxes_wbf_twh.py``` in ```/dfs/data/miniconda/envs/your_env_name/lib/pythonx.x/site-packages/ensemble_boxes/``` <br>
put your tube files into ```./tubes``` <br>
generate ensemble tube files in ```./ensemble_tubes```
### run:
```python ensemble_wbf.py tubes(or your tube path) ensemble_tubes(or your ensemble tube path) tube_file_1.pkl,tube_file_2.pkl,...,tube_file_n.pkl (--dump=map_file_path) (--tuning_iou_thr) (--tuning_skip_box_thr) (tuning_conf_type)```
### example:
```python ensemble_wbf.py tubes ensemble_tubes res50C2D30_test,res50I3D30_test,res50RCLSTM30_test,res50RCN28_test,res50SlowFast30_test test --dump=map.pkl --tuning_iou_thr --tuning_conf_type```

