#!/usr/bin/env python
# coding: utf-8
import argparse

import pickle
import numpy as np
import pandas as pd 
import os
from os import path as osp
import json
import zipfile



def mmdet2roadr(input_path, output_path, topk=20, agent_thres = 0.5):

    df2 = pd.read_pickle(input_path)
    road = dict()


    for i in range(len(df2)):
        dict_frame = dict()

        seq = df2[i]['img_path'].split('/')[-2]
        frame = os.path.basename(df2[i]['img_path'])
        bbox_list = df2[i]['pred_instances']['bboxes'].numpy()
        agent_scores_list = df2[i]['pred_instances']['agent_scores'][:,:-1].numpy()
        act_scores_list = df2[i]['pred_instances']['act_scores'].numpy()
        loc_scores_list = df2[i]['pred_instances']['loc_scores'].numpy()
       
        box_score_list = []
      
        num_agent = agent_scores_list.shape[1]
        num_action = act_scores_list.shape[1]
        
        for j in range(min(len(bbox_list), topk)):
                        
            if all([p < agent_thres for p in agent_scores_list[j]]):
                    continue
            
            scores_list = agent_scores_list[j]
            scores_list = np.append(scores_list,(act_scores_list[j]))
            scores_list = np.append(scores_list,( loc_scores_list[j]))
            frame_dict = dict()
            frame_dict['bbox'] = bbox_list[j]
            frame_dict['labels'] = scores_list
            box_score_list.append(frame_dict)

        dict_frame[frame] = box_score_list
    
        if seq in road :
            road[seq].update(dict_frame) 

        else: 
            road[seq] = (dict_frame)
 
    print(road.keys())


    # with open((osp.join(output_path, 'results_t1.pkl')), 'wb') as preds_f:
    with open(output_path + 'results_t1.pkl', 'wb') as preds_f:
        pickle.dump(road, preds_f)

    
    zf = zipfile.ZipFile(output_path + 'mmdet_t1.zip', 'w', zipfile.ZIP_DEFLATED)
    zf.writestr('mmdet_t1.pkl', pickle.dumps(road))

   


def main():
    parser = argparse.ArgumentParser(description='Converting mmdet ouput format to roadr submission format')
    parser.add_argument('INPUT', default='',
                        type=str, help='mmdet result format')
    parser.add_argument('OUTPUT_dir', default='',
                        type=str, help='Roadr submission format directory')
    parser.add_argument('--topk', default=20,
                        type=int, help='Maximun box output')
    parser.add_argument('--agent_thres', default=0.5,
                        type=int, help='Action threshold')

    args = parser.parse_args()
    print('input: ', args.INPUT)
    print('output: ', args.OUTPUT_dir)
    
    mmdet2roadr(args.INPUT, args.OUTPUT_dir, args.topk, args.agent_thres)

if __name__ == "__main__":
    main()
