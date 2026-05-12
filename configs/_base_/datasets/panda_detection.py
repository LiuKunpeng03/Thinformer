
# dataset settings
dataset_type = 'PandaDataset'
data_root = 'data/PANDA_processed'
file_client_args = dict(backend='disk')

_metainfo = dict(classes=(
    'person',
))
_num_classes = 1

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor'
        )
    )
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=_metainfo,
        data_root=data_root,
        ann_file='train_mix_all_train.json',
        data_prefix=dict(img='patch_mix_alltrain'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        limit_n=0,
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=_metainfo,
        data_root=data_root,
        # ann_file='test_s4.json',
        ann_file='val_mix.json',
        data_prefix=dict(img='patch_mix'),
        test_mode=True,
        pipeline=test_pipeline,
        limit_n=0,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='PANDAMetric',
    # ann_file=f'{data_root}/val_mix.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)
test_evaluator = val_evaluator