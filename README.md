# Road_R Challenge
This repository contains our code for the second task of the ROAD-R Challenge.

|   | Precision@0.5 | Recall@0.5 | F1-score@0.5|
| ------------- | :---: | :---: | :---: |
| PCIE_LR | 0.63 | 0.52 | 0.57 |

Our final results were generated by combining outputs from multiple models, including 6 models from the baseline and 1 model from MMdetection.

## Baseline Model
We followed the baseline from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep) for both data preprocessing and post-processing.
With the default configuration, we trained and obtained results from several backbones, including: resnet50RCGRU, resnet50I3D, resnet50C2D, resnet50RCN-NL, resnet50Slowfast, and resnet101Slowfast, using the baseline repository.
All the generated results are in road-r task 1 format, which allows for further model ensemble.

## MMdetection Model
We utilized the [MMdetection](https://github.com/open-mmlab/mmdetection) to conduct multi-label detection.
Our system is based on MMDetection 3.1.0 with some modification to enable the detector to work with multi-label datasets.
 
### Training
1. Convert Road-R anotation json file to COCO format by running ```road_r2coco.py```
2. Train the faster-rcnn with only the agent for 12 epochs, setting 'with_act' and 'with_loc' to 'False', using the faster-RCNN COCO pre-trained model
3. Train the agent model with action and location labels by setting 'with_act' and 'with_loc' to 'True'
### Testing
1. Generat output.pkl file
2. Convert the output.pkl in mmdetection format to the Road-R task 1 submission format by running the command:
   ```python mmdet2roadr_out.py mmdet_output.pkl save_dir/ --topk 20 --agent_thres 0.5```

 

## Model Ensemble
After generating pkl files with Road-R task 1 format using baseline models and the mmdetection model, we use ensemble.py to produce a submission.pkl file.  
The path to the pkl path from all models are listed in ```ensemble_input_list.txt```.  We then execute the command ```python ensemble.py ensemble_input_list.txt ensemble_road_r_val_pkl/ --topk 20 --skip_box_thr 0.9``` to complete the process
Lastly, we generated a submission file for task 2 by running ```post_processing_raw.py``` from [Road-R](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge#dep) repository.
