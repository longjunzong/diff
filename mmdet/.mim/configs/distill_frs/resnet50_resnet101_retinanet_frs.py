_base_ = [
    '../pascal_voc/retinanet_r50_fpn_1x_voc0712.py'
]

model = dict(
    type='Distilling_FRS_Single',

    distill = dict(
        teacher_cfg='./configs/pascal_voc/retinanet_r101_fpn_1x_voc0712.py',
        teacher_model_path='/home/host/jp/FGD-master/epoch_4.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.005,
        distill_cls_weight=0.02,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]

seed=520
# find_unused_parameters=True
