_base_ = './retinanet_r50_fpn_1x_voc0712.py'
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet18')))
