import numbers
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16
import torch

from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class DanRestorer(BasicRestorer):
    def __init__(self,
                 loop,
                 estimator,
                 kernel_size=21,
                 pca_matrix_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.loop = loop
        self.pca_matrix_path = pca_matrix_path
        self.ksize = kernel_size

        self.estimator = build_backbone(estimator)

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1

        self.register_buffer("init_kernel", kernel)
        init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
            self.encoder
        )[:, 0]
        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lq, gt=None, kernel=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            kernel: Input kernel
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt, kernel)

    def forward_train(self, lq, gt, gt_kernel):
        losses = dict()
        # print("lq:{} , gt:{},  gt_kernel:{}".
        #       format(torch.sum(torch.isnan(lq)), torch.sum(torch.isnan(gt)), torch.sum(torch.isnan(gt_kernel))))
        # losses['loss_ker'] = []
        B, C, H, W = lq.shape
        ker_map = self.init_ker_map.repeat([B, 1])
        self.encoder = self.encoder.to(gt.device)

        for i in range(self.loop):
            sr = self.generator(lq, ker_map.detach())
            kernel = self.estimator(sr.detach(), lq)
            ker_map = kernel.view(B, 1, self.ksize ** 2).matmul(self.encoder)[:, 0]

        losses['loss_pix_img'] = self.pixel_loss(sr, gt)
        losses['loss_pix_kernel'] = self.pixel_loss(kernel, gt_kernel.view(*kernel.shape))
        # print('loss_kernel:{}'.format(losses['loss_pix_kernel']))



        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """

        ker_map = self.init_ker_map.repeat([lq.shape[0], 1])

        output = self.generator(lq, ker_map)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

