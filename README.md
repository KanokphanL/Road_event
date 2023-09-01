# Road_event
ensemble tubes:
	put ensemble_boxes_wbf_twh.py in /dfs/data/miniconda/envs/your_env_name/lib/pythonx.x/site-packages/ensemble_boxes/
	put your tube files into ./tubes
	generate ensemble tube files in ./ensemble_tubes
	run:
		pythom ensemble_wbf.py tubes(or your tube path) ensemble_tubes(or your ensemble tube path) tube_file_1.pkl,tube_file_2.pkl,...,tube_file_n.pkl (--dump=map_file_path) (--tuning_iou_thr) (--tuning_skip_box_thr) (tuning_conf_type)
	example:
		python ensemble_wbf.py tubes ensemble_tubes res50C2D30_test,res50I3D30_test,res50RCLSTM30_test,res50RCN28_test,res50SlowFast30_test test --dump=map.pkl --tuning_iou_thr --tuning_conf_type

