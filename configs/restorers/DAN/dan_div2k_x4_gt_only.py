exp_name = 'DAN_DIV2k_x4_GT_ONLY'
pca_matrix_path = '/data/xieyufeng/dan_mmediting/tools/data/blind-super-resolution/div2k/pca_matrix.pth'
# model settings
estimator = dict(type='DanKerEstimator',
                 in_nc=3,
                 nf=64,
                 num_blocks=5,
                 scale=4,
                 kernel_size=31)

generator = dict(type='DanSRGenerator',
                 in_nc=3,
                 nf=64,
                 num_blocks=40,
                 scale=4,
                 input_para=10)

model = dict(
    type='DanRestorer',
    loop=8,
    estimator=estimator,
    generator=generator,
    kernel_size=31,
    pca_matrix_path=pca_matrix_path,
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
)
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=8, convert_to='y')

# config for blur kernel
kernel_options={
    'random':True,
    'kernel_size':31,
    'sigma_min':0.5,
    'sigma_max':6.0,
    'isotropic': False,
    'random_disturb': True
}

# dataset settings
train_dataset_type = 'SRFolderGTDataset'
val_dataset_type = 'BSRTestFolderDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadKernelFromFile',
        key='kernel'),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(type='Degradation', keys=['gt'], opt=kernel_options),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'kernel'], to_float32=True),
    dict(type='Collect', keys=['lq', 'gt', 'kernel'], meta_keys=['gt_path'])
]

val_pipeline = [
    # dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadKernelFromFile',
        key='kernel'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'kernel']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    # dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    # dict(
    #     type='LoadKernelFromFile',
    #     key='kernel'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True, pin_memory=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1, drop_last=True, pin_memory=False),
    test_dataloader=dict(samples_per_gpu=1, drop_last=True, pin_memory=False),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='/data/xieyufeng/DIV2K/DIV2K_train_HR_sub',
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder='/data/xieyufeng/DIV2K/DIV2KRK/lr_x4',
        gt_folder='/data/xieyufeng/DIV2K/DIV2KRK/gt',
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='/data/xieyufeng/DIV2K/DIV2KRK/lr_x4',
        gt_folder='/data/xieyufeng/DIV2K/DIV2KRK/gt',
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 400000
# lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=0.5, min=1.25e-05)
lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=0.5, min_lr=1.25e-05)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=100, save_image=False, gpu_collect=True)
evaluation = dict(interval=1000, save_image=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
# load_from = './experiments/tdan_vimeo90k_bdx4_lr1e-4_400k/iter_400000.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]