from torch import nn

from mmedit.models.registry import BACKBONES
import torch
from mmcv.runner import load_checkpoint

from mmedit.utils import get_root_logger

from mmedit.models.common import (ResidualBlockWithDropout,
                                  generation_init_weights)
class DPCB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=1):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels=nf1, out_channels=nf1, kernel_size=ksize1, stride=1, padding=ksize1 // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=nf1, out_channels=nf1, kernel_size=ksize1, stride=1, padding=ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels=nf2, out_channels=nf1, kernel_size=ksize2, stride=1, padding=ksize2 // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=nf1, out_channels=nf1, kernel_size=ksize2, stride=1, padding=ksize2 // 2),
        )

    def forward(self, x):
        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        x[0] = x[0] + torch.mul(f1, f2)
        x[1] = x[1] + f2
        return x


class DPCG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        """

        Args:
            nf1:
            nf2:
            ksize1:
            ksize2:
            nb:  nums of DPCBs
        """
        super().__init__()

        self.body = nn.Sequential(*[DPCB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


@BACKBONES.register_module()
class DanSRGenerator(nn.Module):
    def __init__(
            self,
            nf=64,
            num_blocks=16,
            ng=5,
            in_nc=3,
            scale=4,
            input_para=10,
            min=0.0,
            max=1.0,
            init_cfg=None
    ):
        super(DanSRGenerator, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = num_blocks

        out_nc = in_nc

        self.head1 = nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.head2 = nn.Conv2d(in_channels=input_para, out_channels=nf, kernel_size=1, stride=1, padding=0)

        body = [DPCG(nf, nf, 3, 1, num_blocks) for _ in range(ng)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, input, ker_code):
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1))

        f1 = self.head1(input)
        f2 = self.head2(ker_code_exp)
        inputs = [f1, f2]
        f, _ = self.body(inputs)
        f = self.fusion(f)
        out = self.upscale(f)

        return out  # torch.clamp(out, min=self.min, max=self.max)

    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')


@BACKBONES.register_module()
class DanKerEstimator(nn.Module):
    def __init__(
            self,
            in_nc=1,
            nf=64,
            num_blocks=5,
            scale=4,
            kernel_size=4,
            init_cfg=None
    ):
        super(DanKerEstimator, self).__init__()

        self.ksize = kernel_size

        self.head_LR = nn.Sequential(
            # CenterCrop(self.ksize + scale),
            nn.Conv2d(in_nc, nf // 2, 5, 1, 2)
        )
        self.head_HR = nn.Sequential(
            # CenterCrop(self.ksize + scale),
            nn.Conv2d(in_nc, nf // 2, scale * 4 + 1, scale, scale * 2),
        )

        self.body = DPCG(nf // 2, nf // 2, 3, 3, num_blocks)

        self.tail = nn.Sequential(
            nn.Conv2d(nf // 2, nf, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, self.ksize ** 2, 1, 1, 0),
            nn.Softmax(1),
        )
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, GT, LR):
        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)

        f = [lrf, hrf]
        f, _ = self.body(f)
        f = self.tail(f)

        return f.view(*f.size()[:2])



    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')