# 40 CPUs required

from ensemble_boxes import *

import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import json
import pickle

from modules.evaluation import get_gt_tubes, get_det_class_tubes, get_gt_class_tubes, compute_class_ap
from modules.tube_helper import get_tube_3Diou

import os
import argparse
from typing import Optional

def parse_args():
    parser = argparse.ArgumentParser(description = 'ensemble boxes')
    parser.add_argument('tube_path', help = 'tubes')
    parser.add_argument('ensemble_tube_path', help = 'ensemble_tubes')
    parser.add_argument('tube_name', help = 'tube name 1_tube name 2_..._tube name n]')
    parser.add_argument('type', help = 'val or test')
    parser.add_argument('--tuning_iou_thr', action='store_true')
    parser.add_argument('--tuning_skip_box_thr', action='store_true')
    parser.add_argument('--tuning_conf_type', action='store_true')
    parser.add_argument('--dump_path', default = None)
    args = parser.parse_args()
    return args

def evaluate_tubes(anno_file, det_file,  subset='val', dataset='road_waymo', iou_thresh=0.1, metric_type='stiou'):

    print('Evaluating tubes for datasets '+ dataset)
    print('GT FILE:: '+ anno_file)
    print('Result File:: '+ det_file)
    used_labels = {"agent_labels": ["Ped", "Car", "Cyc", "Mobike", "SmalVeh", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"],
                       "action_labels": ["Red", "Amber", "Green", "MovAway", "MovTow", "Mov", "Rev", "Brake", "Stop", "IncatLft", "IncatRht", "HazLit", "TurLft", "TurRht", "MovRht", "MovLft", "Ovtak", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
                       "loc_labels": ["VehLane", "OutgoLane", "OutgoCycLane", "OutgoBusLane", "IncomLane", "IncomCycLane", "IncomBusLane", "Pav", "LftPav", "RhtPav", "Jun", "xing", "BusStop", "parking", "LftParking", "rightParking"],
                       "duplex_labels": ["Ped-MovAway", "Ped-MovTow", "Ped-Mov", "Ped-Stop", "Ped-Wait2X", "Ped-XingFmLft", "Ped-XingFmRht", "Ped-Xing", "Ped-PushObj", "Car-MovAway", "Car-MovTow", "Car-Brake", "Car-Stop", "Car-IncatLft", "Car-IncatRht", "Car-HazLit", "Car-TurLft", "Car-TurRht", "Car-MovRht", "Car-MovLft", "Car-XingFmLft", "Car-XingFmRht", "Cyc-MovAway", "Cyc-MovTow", "Cyc-Stop", "Mobike-Stop", "MedVeh-MovAway", "MedVeh-MovTow", "MedVeh-Brake", "MedVeh-Stop", "MedVeh-IncatLft", "MedVeh-IncatRht", "MedVeh-HazLit", "MedVeh-TurRht", "MedVeh-XingFmLft", "MedVeh-XingFmRht", "LarVeh-MovAway", "LarVeh-MovTow", "LarVeh-Stop", "LarVeh-HazLit", "Bus-MovAway", "Bus-MovTow", "Bus-Brake", "Bus-Stop", "Bus-HazLit", "EmVeh-Stop", "TL-Red", "TL-Amber", "TL-Green"], 
                       "triplet_labels": ["Ped-MovAway-LftPav", "Ped-MovAway-RhtPav", "Ped-MovAway-Jun", "Ped-MovTow-LftPav", "Ped-MovTow-RhtPav", "Ped-MovTow-Jun", "Ped-Mov-OutgoLane", "Ped-Mov-Pav", "Ped-Mov-RhtPav", "Ped-Stop-OutgoLane", "Ped-Stop-Pav", "Ped-Stop-LftPav", "Ped-Stop-RhtPav", "Ped-Stop-BusStop", "Ped-Wait2X-RhtPav", "Ped-Wait2X-Jun", "Ped-XingFmLft-Jun", "Ped-XingFmRht-Jun", "Ped-XingFmRht-xing", "Ped-Xing-Jun", "Ped-PushObj-LftPav", "Ped-PushObj-RhtPav", "Car-MovAway-VehLane", "Car-MovAway-OutgoLane", "Car-MovAway-Jun", "Car-MovTow-VehLane", "Car-MovTow-IncomLane", "Car-MovTow-Jun", "Car-Brake-VehLane", "Car-Brake-OutgoLane", "Car-Brake-Jun", "Car-Stop-VehLane", "Car-Stop-OutgoLane", "Car-Stop-IncomLane", "Car-Stop-Jun", "Car-Stop-parking", "Car-IncatLft-VehLane", "Car-IncatLft-OutgoLane", "Car-IncatLft-IncomLane", "Car-IncatLft-Jun", "Car-IncatRht-VehLane", "Car-IncatRht-OutgoLane", "Car-IncatRht-IncomLane", "Car-IncatRht-Jun", "Car-HazLit-IncomLane", "Car-TurLft-VehLane", "Car-TurLft-Jun", "Car-TurRht-Jun", "Car-MovRht-OutgoLane", "Car-MovLft-VehLane", "Car-MovLft-OutgoLane", "Car-XingFmLft-Jun", "Car-XingFmRht-Jun", "Cyc-MovAway-OutgoCycLane", "Cyc-MovAway-RhtPav", "Cyc-MovTow-IncomLane", "Cyc-MovTow-RhtPav", "MedVeh-MovAway-VehLane", "MedVeh-MovAway-OutgoLane", "MedVeh-MovAway-Jun", "MedVeh-MovTow-IncomLane", "MedVeh-MovTow-Jun", "MedVeh-Brake-VehLane", "MedVeh-Brake-OutgoLane", "MedVeh-Brake-Jun", "MedVeh-Stop-VehLane", "MedVeh-Stop-OutgoLane", "MedVeh-Stop-IncomLane", "MedVeh-Stop-Jun", "MedVeh-Stop-parking", "MedVeh-IncatLft-IncomLane", "MedVeh-IncatRht-Jun", "MedVeh-TurRht-Jun", "MedVeh-XingFmLft-Jun", "MedVeh-XingFmRht-Jun", "LarVeh-MovAway-VehLane", "LarVeh-MovTow-IncomLane", "LarVeh-Stop-VehLane", "LarVeh-Stop-Jun", "Bus-MovAway-OutgoLane", "Bus-MovTow-IncomLane", "Bus-Stop-VehLane", "Bus-Stop-OutgoLane", "Bus-Stop-IncomLane", "Bus-Stop-Jun", "Bus-HazLit-OutgoLane"]}

    if dataset == 'road' or dataset == 'roadpp' or dataset=='road_waymo':
        with open(anno_file, 'r') as fff:
            final_annots = json.load(fff)
    else:
        with open(anno_file, 'rb') as fff:
            final_annots = pickle.load(fff)
    
    with open(det_file, 'rb') as fff:
        detections = pickle.load(fff)

    if dataset == 'road' or dataset =='roadpp' or dataset=='road_waymo':
        label_types = final_annots['label_types']
    else:
        label_types = ['action']
    
    results = {} 
    for _, label_type in enumerate(label_types):

        if dataset != 'road' and dataset != 'roadpp' and dataset!= 'road_waymo' :
            classes = final_annots['classes']
        else:
            classes = used_labels[label_type+'_labels']

        print('Evaluating {} {}'.format(label_type, len(classes)))
        ap_all = []
        re_all = []
        ap_strs = []
        sap = 0.0
        
        video_list, gt_tubes = get_gt_tubes(final_annots, subset, label_type, dataset)
        det_tubes = {}
        
        for videoname in video_list:
            det_tubes[videoname] = detections[label_type][videoname]

        for cl_id, class_name in enumerate(classes):
            
            class_dets = get_det_class_tubes(det_tubes, cl_id)
            class_gts = get_gt_class_tubes(gt_tubes, cl_id)
            
            class_ap, num_postives, count, recall = compute_class_ap(class_dets, class_gts, get_tube_3Diou, iou_thresh, metric_type=metric_type)
            
            recall = recall*100
            sap += class_ap
            ap_all.append(class_ap)
            re_all.append(recall)
            ap_str = class_name + ' : ' + str(num_postives) + \
                ' : ' + str(count) + ' : ' + str(class_ap) +\
                ' : ' + str(recall)
            ap_strs.append(ap_str)
        mAP = sap/len(classes)
        mean_recall = np.mean(np.asarray(re_all))
        ap_strs.append('\nMean AP:: {:0.2f} mean Recall {:0.2f}'.format(mAP,mean_recall))
        results[label_type] = {'mAP':mAP, 'ap_all':ap_all, 'ap_strs':ap_strs, 'recalls':re_all, 'mR':mean_recall}
        print('MAP:: {}'.format(mAP))
    
    return results    

def evalai_result(args, anno_file, ensemble_tube_name, arg):  
    
    agent_map = []
    event_map = []
    
    results = evaluate_tubes(anno_file = anno_file,
                        det_file = os.path.join(args.ensemble_tube_path, '{}_{}_{}_{}.pkl'.format(ensemble_tube_name, arg[0], arg[1], arg[2])),
                        subset = args.type,
                        dataset = 'road_waymo',
                        iou_thresh = 0.1,
                        metric_type = 'stiou')
    agent_map.append(results['agent']['mAP'])
    event_map.append(results['triplet']['mAP'])
    
    results = evaluate_tubes(anno_file = anno_file,
                        det_file = os.path.join(args.ensemble_tube_path, '{}_{}_{}_{}.pkl'.format(ensemble_tube_name, arg[0], arg[1], arg[2])),
                        subset = args.type,
                        dataset = 'road_waymo',
                        iou_thresh = 0.2,
                        metric_type = 'stiou')
    agent_map.append(results['agent']['mAP'])
    event_map.append(results['triplet']['mAP'])
    
    results = evaluate_tubes(anno_file = anno_file,
                        det_file = os.path.join(args.ensemble_tube_path, '{}_{}_{}_{}.pkl'.format(ensemble_tube_name, arg[0], arg[1], arg[2])),
                        subset = args.type,
                        dataset='road_waymo',
                        iou_thresh=0.5,
                        metric_type='stiou')
    agent_map.append(results['agent']['mAP'])
    event_map.append(results['triplet']['mAP'])
    
    mean_agent_map = np.mean(agent_map)
    mean_event_map = np.mean(event_map)
    agent_map.append(mean_agent_map)
    event_map.append(mean_event_map)
    
    return agent_map, event_map

def tuning(args):
    maps = []
        
    pkls_list = args.tube_name.split(',')
    
    ensemble_tube_name = '_'.join(pkls_list)
    
    tubes_list = []
    for pkl in pkls_list:
        with open(os.path.join(args.tube_path, pkl + '.pkl'), 'rb') as f:
            tubes_list.append(pickle.load(f))
    
    assert args.type in ['val', 'test'], 'type must be val or test'
    if args.type == 'val':
        anno_file = '/dfs/data/roadpp/road_waymo/road_waymo_trainval_v1.0.json'
    else:
        anno_file = '/dfs/data/roadpp_test/road_waymo/road_waymo_test_v1.0.json'
    
    all_args = []
    its = [0.2]
    sbts = [0]
    cts = ['avg']
    if args.tuning_iou_thr:
        its = np.arange(0.1, 0.4, 0.1)
    if args.tuning_skip_box_thr:
        sbts = np.arange(0, 0.05, 0.01)
    if args.tuning_conf_type:
        cts = ['avg', 'max']
    for it in its:
        for sbt in sbts:
            for ct in cts:
                all_args.append([it, sbt, ct])
    
    for arg in all_args:
    
        ensemble_tubes = {}

        for t in tqdm(tubes_list[0].keys()):
            ensemble_tubes[t] = Parallel(n_jobs = 40)(delayed
                            (weighted_boxes_fusion_twh)(
                            video_name = v,
                            tubes_list = [tubes[t][v] for tubes in tubes_list], 
                            weights = None, 
                            iou_thr = arg[0], 
                            skip_box_thr = arg[1], 
                            conf_type = arg[2], 
                            allows_overflow = False)
                     for v in tqdm(tubes_list[0][t].keys()))
            ensemble_tubes[t] = dict(ensemble_tubes[t])

        with open(os.path.join(args.ensemble_tube_path, '{}_{}_{}_{}.pkl'.format(ensemble_tube_name, arg[0], arg[1], arg[2])), 'wb') as f:
            pickle.dump(ensemble_tubes, f)

        agent_map, event_map = evalai_result(args, anno_file, ensemble_tube_name, arg)

        print({'arg': arg, 'agent mAP': agent_map, 'event mAP': event_map})
        maps.append({'arg': arg, 'agent mAP': agent_map, 'event mAP': event_map})
        
    return maps

def main():
    args = parse_args()
    maps = tuning(args)
    if args.dump_path is not None:
        with open(args.dump_path, 'wb') as f:
            pickle.dump(maps, f)

if __name__ == '__main__':
    main()
