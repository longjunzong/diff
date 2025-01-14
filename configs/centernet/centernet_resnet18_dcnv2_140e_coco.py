_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    #neck=dict(
    #    type='CTResNetNeck',
    #    in_channel=512,
    #    num_deconv_filters=(256, 128, 64),
    #    num_deconv_kernels=(4, 4, 4),
    #    use_dcn=True),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='GaussianFocalLoss1',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg = dict(max_epochs=12),
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))


#image_size = (1024, 1024)
#batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
#
#model = dict(
#    type='CenterNet',
#    backbone=dict(
#        type='ResNet',
#        depth=50,
#        num_stages=4,
#        out_indices=(0, 1, 2, 3),
#        frozen_stages=1,
#        norm_cfg=dict(type='BN', requires_grad=True),
#        norm_eval=True,
#        style='pytorch',
#        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#    neck=dict(
#        type='FPN',
#        in_channels=[256, 512, 1024, 2048],
#        out_channels=256,
#        start_level=1,
#        add_extra_convs='on_output',
#        num_outs=5,
#        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
#        relu_before_extra_convs=True),
#    bbox_head=dict(
#        type='CenterNetUpdateHead',
#        num_classes=80,
#        in_channels=256,
#        stacked_convs=4,
#        feat_channels=256,
#        strides=[8, 16, 32, 64, 128],
#        loss_cls=dict(
#            type='GaussianFocalLoss1',
#            pos_weight=0.25,
#            neg_weight=0.75,
#            loss_weight=1.0),
#        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
#    ),
#    train_cfg=None,
#    test_cfg=dict(
#        nms_pre=1000,
#        min_bbox_size=0,
#        score_thr=0.05,
#        nms=dict(type='nms', iou_threshold=0.6),
#        max_per_img=100))

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
data_root = '/home/host/7T_Disk/TrackingDataset/track_train_dataset/COCO/'

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=1,            #bs
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[9, 12])  # the real step is [18*5, 24*5]
runner = dict(max_epochs=12)  # the real epoch is 28*5=140

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=2) #bs
