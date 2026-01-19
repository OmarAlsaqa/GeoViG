_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='geovig_m_feat',
        style='pytorch',
        pretrained=False,
        use_detect_adapter=True,
        init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 384],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# Dataset settings
dataset_type = 'KvasirDataset'
data_root = '../../data/Kvasir-SEG'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='LoadMasksFromAnn'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + '/kavsir_bboxes.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + '/kavsir_bboxes.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + '/kavsir_bboxes.json',
        pipeline=test_pipeline))

# Fine-tuning config
load_from = 'geovig_m_seg_4G/epoch_12.pth'

evaluation = dict(metric='mAP')

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2))
