"""Zero-DCE++ 네트워크 - 저조도 영상 밝기/대비 보정
   공식 저장소(Li-Chongyi/Zero-DCE_extension)의 enhance_net_nopool 구조와
   파라미터 이름까지 정확히 일치시켜 사전학습 가중치(Epoch99.pth)를 그대로 로드.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """Depthwise Separable Conv (공식 구현과 동일한 속성명/bias 설정)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1,
                                    padding=1, groups=in_ch, bias=True)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                                    padding=0, bias=True)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEPlus(nn.Module):
    """Zero-DCE++ 본체 - 공식 enhance_net_nopool 과 동일 구조.
       num_iter 는 학습 시점과 동일한 8 (LE-curve 반복 적용 횟수)
    """
    def __init__(self, scale_factor: int = 12):
        super().__init__()
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        n = 32
        self.e_conv1 = CSDN_Tem(3, n)
        self.e_conv2 = CSDN_Tem(n, n)
        self.e_conv3 = CSDN_Tem(n, n)
        self.e_conv4 = CSDN_Tem(n, n)
        self.e_conv5 = CSDN_Tem(n * 2, n)
        self.e_conv6 = CSDN_Tem(n * 2, n)
        self.e_conv7 = CSDN_Tem(n * 2, 3)

    @staticmethod
    def _enhance(x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        # 공식 구현과 동일: LE-curve 를 정확히 8회 적용
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor,
                                   mode="bilinear", align_corners=False)
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        return self._enhance(x, x_r)
