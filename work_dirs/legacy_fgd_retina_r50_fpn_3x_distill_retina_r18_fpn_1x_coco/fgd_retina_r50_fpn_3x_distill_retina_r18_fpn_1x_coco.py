dataset_type = 'CocoFmtDataset'
data_root = 'data/tiny_set/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=
        'data/tiny_set/mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
        img_prefix='data/tiny_set/erase_with_uncertain_dataset/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='ScaleMatchResize',
                scale_match_type='ScaleMatch',
                anno_file=
                'data/tiny_set/mini_annotations/tiny_set_train_all_erase.json',
                bins=100,
                default_scale=0.25,
                scale_range=(0.1, 1)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/tiny_set/mini_annotations/tiny_set_test_all.json',
        img_prefix='data/tiny_set/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(333, 200),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/tiny_set/mini_annotations/tiny_set_test_all.json',
        img_prefix='data/tiny_set/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(333, 200),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.01
beta_fgd = 1
gamma_fgd = 1e-05
lambda_fgd = 1
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained=
    '/home/host/jp/mm-new/mmdetection/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth',
    init_student=True,
    distill_cfg=[
        dict(
            student_module='neck.fpn_convs.4.conv',
            teacher_module='neck.fpn_convs.4.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_4',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.01,
                    beta_fgd=1,
                    gamma_fgd=1e-05,
                    lambda_fgd=1)
            ]),
        dict(
            student_module='neck.fpn_convs.3.conv',
            teacher_module='neck.fpn_convs.3.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_3',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.01,
                    beta_fgd=1,
                    gamma_fgd=1e-05,
                    lambda_fgd=1)
            ]),
        dict(
            student_module='neck.fpn_convs.2.conv',
            teacher_module='neck.fpn_convs.2.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_2',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.01,
                    beta_fgd=1,
                    gamma_fgd=1e-05,
                    lambda_fgd=1)
            ]),
        dict(
            student_module='neck.fpn_convs.1.conv',
            teacher_module='neck.fpn_convs.1.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_1',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.01,
                    beta_fgd=1,
                    gamma_fgd=1e-05,
                    lambda_fgd=1)
            ]),
        dict(
            student_module='neck.fpn_convs.0.conv',
            teacher_module='neck.fpn_convs.0.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_0',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.01,
                    beta_fgd=1,
                    gamma_fgd=1e-05,
                    lambda_fgd=1)
            ])
    ])
student_cfg = 'configs/retinanet/retinanet_r18_fpn_1x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
work_dir = './work_dirs/fgd_retina_r50_fpn_3x_distill_retina_r18_fpn_1x_coco'
auto_resume = False
gpu_ids = [0]
