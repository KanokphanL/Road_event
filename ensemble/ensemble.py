import os
import pickle
from tqdm import tqdm

import numpy as np
from utils.ensemble_boxes_wbf import weighted_boxes_fusion
import zipfile
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description = 'ensemble boxes')
    parser.add_argument('pkl_list', help = 'path_to_pkl_list.txt')
    parser.add_argument('ensemble_tube_path', help = 'ensemble_tubes')
    parser.add_argument('--topk', default = 20, type=float)
    parser.add_argument('--iou_thr', default = 0.2, type=float)
    parser.add_argument('--skip_box_thr', default = 0, type=float)
    parser.add_argument('--conf_type', default = 'avg', help = 'avg or max')
    args = parser.parse_args()
    return args

def ensemble(args):
    file_list = []
    file1 = open(args.pkl_list,"r")
    i = 0
    for line in file1:
        if i == 0:
            ensemble_file_name = line[:-1] + '.pkl'
            ensemble_zip_name = line[:-1] + '.zip'
            print('ensemble_file_name: ', ensemble_file_name)
        else:
            line = line[:-1]
            print(line)
            with open(line, 'rb') as f:        
                file = pickle.load(f)
                file_list.append(file)
        i = i+1
    
    ensemble_file = {}
    for v in tqdm(file_list[0].keys(), position = 0):
        ensemble_file[v] = {}
        for f in tqdm(file_list[0][v].keys(), position = 0):
            boxes_list = []
            scores_list = []
            all_scores_list = []
            labels_list = []
            i = 0
            for file in file_list:
                i = i+1
                if f in file[v].keys():
                    a = np.array([b['labels'][:10].max() for b in file[v][f]])
                    boxes_list.append(np.array([b['bbox'] for b in file[v][f]]))
                    scores_list.append(np.array([b['labels'][:10].max() for b in file[v][f]]))
                    all_scores_list.append(np.array([b['labels'] for b in file[v][f]]))
                    labels_list.append(np.array([b['labels'][:10].argmax() for b in file[v][f]]))
                  
            boxes, scores, all_scores, labels = weighted_boxes_fusion(boxes_list,
                                                                              scores_list,
                                                                              all_scores_list,
                                                                              labels_list,
                                                                              iou_thr=args.iou_thr,
                                                                              skip_box_thr=args.skip_box_thr,
                                                                              conf_type=args.conf_type) 
            assert len(boxes) == len(all_scores)
            ensemble_file[v][f] = []
            for i in range(min(len(boxes), int(args.topk))):
                ensemble_file[v][f].append({'bbox': boxes[i], 'labels': all_scores[i]})
             
    if not os.path.exists(args.ensemble_tube_path):
        os.mkdir(args.ensemble_tube_path)
        
    with open(args.ensemble_tube_path + ensemble_file_name, 'wb') as f:
        pickle.dump(ensemble_file, f)
        
    zf = zipfile.ZipFile(args.ensemble_tube_path + ensemble_zip_name, 'w', zipfile.ZIP_DEFLATED)
    zf.writestr(ensemble_file_name, pickle.dumps(ensemble_file))


def main():
    args = parse_args()
    ensemble(args)

if __name__ == '__main__':
    main()
