_base_ = [
     '../../configs/_base_/models/faster-rcnn_r50_fpn.py', 
    './road-r_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]
work_dir = 'work_dirs/multi_label_no_flip_task2'
# model settings
model = dict(
    type='RoadrMultilabelTwoStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # freeze the first backbone stage, set to -1 means not freeze
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,   # freeze the BN layers' mean&std, set to False means not freeze
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='RoadRStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RoadRConvFCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10, # for road_r
            num_action_classes=19,
            num_location_classes=12,
            with_cls=True,
            with_act=True,
            with_loc=True,

            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_action=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_location=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
)


# dataset settings
train_dataloader = dict(
    batch_size=8, 
    num_workers=2, 
    )

test_dataloader = dict(
    dataset=dict(
#         ann_file='road_r_val_coco.json',
        ann_file='road_r_test_coco.json',
        ))

test_evaluator = dict(
    format_only=False, #False, # True,
    )

max_epochs = 24
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=50  # 50
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,#12
        by_epoch=True,
        milestones=[16, 22],
  #      milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD',
                   lr=0.002, # 0.02,
                   momentum=0.9,
                   weight_decay=0.0001),

)

auto_scale_lr = dict(enable=False, base_batch_size=16)

# visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


load_from = 'ckpts/faster_rcnn_r50_lr5e-4_task2_12e.pth'
