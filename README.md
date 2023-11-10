# Road_R Challenge
This repository contains our code for the second task of the ROAD-R Challenge.

|   | Precision@0.5 | Recall@0.5 | F1-score@0.5|
| ------------- | :---: | :---: | :---: |
| Ours | 0.63 | 0.52 | 0.57 |

Our results was ensembled from several model including 5 models from basline and 1 model from mmdection.

## Baseline Model
We followed the baseline from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep) for data pre and post processing.
From the baseone repository trained and generated results from several backbones including: resnet50RCGRU, resnet50I3D, resnet50C2D, resnet50RCN-NL, resnet50Slowfast, and resnet101Slowfast.
All the generated results are in road-r task 1 format for further model ensemble. 

## MMdetection Model
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

## Model Ensemble
After generating .pkl file with road-r task 1 format from baseline models and mmdetection model, we used ensemble.py to generate a submission.pkl file.  
The .pkl path from all models is listed in ```ensemble_input_list.txt```.  Then run ```python ensemble.py ensemble_input_list.txt ensemble_road_r_val_pkl/ --topk 20 --skip_box_thr 0.9```
Finally, we generated a submission file for task 2 by running post_processing_raw.py from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep)

