_base_ = [
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py',
    '../configs/_base_/datasets/whuoptsar_RGBSAR_512x512.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='RGBXDataPreProcessor',
    size=crop_size,
    mean=[80.475, 98.217, 106.364, 53.837, 53.837, 53.837],
    std=[29.48, 26.953, 24.829, 48.475, 48.475, 48.475],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
class_weight = [0.9183, 1.0143, 1.0057, 0.9574, 0.9175, 1.1098, 1.0770]
checkpoint_file = r'pretrains/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth'  
model = dict(
    type='EarlyFusionSegmentorPenalty',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PADPenalty',
        backbone=dict(
            type='mmpretrain.ConvNeXt',
            arch='large',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=0.,  
            gap_before_final_norm=False,
            use_grn=True,  
            init_cfg=dict(type='Pretrained',
                          checkpoint=checkpoint_file,
                          prefix='backbone.')
        ),
        ffm_cfg=dict(type='PAD'),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                 use_sigmoid=False,
                 avg_non_ignore=True,
                 loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         avg_non_ignore=True,
                         loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(_delete_=True,
                     type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.0001,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.05)
                    )

param_scheduler = [
    dict(type='CosineAnnealingLR',
         T_max=80000,
         eta_min=0.0,
         by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(save_best='mIoU', save_last=False, max_keep_ckpts=1))

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(vis_backends=vis_backends)

# find_unused_parameters=True
