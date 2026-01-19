
_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='geovig_b_feat',
        style='pytorch',
        pretrained='../../geovig_b_5e4_8G_300_82_38/checkpoint.pth',
        use_detect_adapter=True, # Enable new adapter
        init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[80, 160, 320, 512],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True) # GroupNorm Stability
    ))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05) # Lower LR + AdamW
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2)) # Gradient Clipping