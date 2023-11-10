
dataset_type = 'RoadRAgentDataset'
data_root = 'data/road_r/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadMultiClassAnnotations', with_bbox=True, multi_label=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackMultiDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 
                   'gt_bboxes', 'gt_bboxes_labels', 'gt_agent_labels', 
                   'gt_agent_labels', 'gt_action_labels', 'gt_location_labels'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadMultiClassAnnotations', with_bbox=True, multi_label=True),

    dict(type='PackMultiDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')
                   )
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='road_r_task-2_train_coco.json',
        data_prefix=dict(img='rgb-images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='road_r_test_coco.json',
        data_prefix=dict(img='rgb-images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'road_r_test_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='road_r_test_coco.json',
        data_prefix=dict(img='rgb-images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'road_r_test_coco.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./res/faster_rcnn/epoch_1')

