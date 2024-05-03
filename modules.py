import torch
from torch import nn


class DepthWiseDilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(DepthWiseDilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            # depth-wise
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
                      bias=False),
            # point-wize linear
            nn.Conv2d(channels, channels, 1, 1, 0, 1, 1, bias=False),
            nn.GroupNorm(num_groups=channels, num_channels=channels),
        )
        self.selu = nn.SELU(inplace=True)
        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
                      bias=False),
            # pw-linear
            nn.Conv2d(channels, channels, 1, 1, 0, 1, 1, bias=False),
            nn.GroupNorm(num_groups=channels, num_channels=channels),
        )

    def forward(self, x):
        res = self.conv0(x)
        res = self.selu(res)
        res = self.conv1(res)
        output = self.selu(res + x)
        return output


class DepthGuidedAttentionBlock(nn.Module):
    def __init__(self, in_channels, dk, dv):
        super(DepthGuidedAttentionBlock, self).__init__()
        # self.w_q = nn.Conv1d(1, dk, kernel_size=1)
        # self.w_k = nn.Conv1d(in_channels, dk, kernel_size=1)
        # self.w_v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.w_q = nn.Conv1d(1, dk, kernel_size=3, padding=1, stride=1)
        self.w_k = nn.Conv1d(in_channels, dk, kernel_size=3, padding=1, stride=1)
        self.w_v = nn.Conv1d(in_channels, dv, kernel_size=3, padding=1, stride=1)

        # 1 / 8 down sample
        self.conv_d = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True)
        )

        # 1 / 4 down sample
        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.SELU(inplace=True)
        )

        # 1 / 4 up sample and dv -> c
        self.conv_v = nn.Sequential(
            nn.ConvTranspose2d(dv, dv, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(dv, dv, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(dv, in_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.SELU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_ori, depth, dk_sqrt):
        # q
        # depth = F.interpolate(depth, scale_factor=0.125, mode='bilinear', align_corners=True)
        depth = self.conv_d(depth)
        b1, c, h1, w1 = depth.shape
        depth = depth.view(b1, c, h1 * w1)
        # [b1, 1, h1 * w1] -> [b1, dk, h1 * w1] -> [b1, h1 * w1, dk]
        q = self.w_q(depth).permute(0, 2, 1)

        # k
        # feature = F.interpolate(feature_ori, scale_factor=0.25, mode='bilinear', align_corners=True)
        feature = self.conv_f(feature_ori)
        b2, n, h2, w2 = feature.shape
        feature = feature.view(b2, n, h2 * w2)
        # [b2,n , h2 * w2] -> [b2, dk, h2 * w2]
        k = self.w_k(feature)

        # v  [b2, n, h2 * w2] -> [b2, dv, h2 * w2] ->  [b2, h2 * w2, dv]
        v = self.w_v(feature).permute(0, 2, 1)
        d_v = v.shape[-1]

        # d_k = torch.tensor(q.shape[-1])

        # attention : [b1, h1 * w1, dk] * [b2, dk, h2 * w2] = [b, h1 * w1, h2 * w2]
        attention = self.softmax(torch.bmm(q, k) / dk_sqrt)

        # result : [b, h1 * w1, dv] -> [b, dv, h1 * w1]
        result = torch.bmm(attention, v).permute(0, 2, 1)
        result = result.view(b1, d_v, h1, w1)
        # result = F.interpolate(result, scale_factor=4, mode='bilinear', align_corners=True)
        result = self.conv_v(result)

        return result + feature_ori


class dense_layer(nn.Module):
    # single dense layer
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(dense_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.selu = nn.SELU()

    def forward(self, x):
        output = self.selu(self.conv(x))
        return torch.cat((x, output), 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, original_channels, num_dense):
        super(ResidualDenseBlock, self).__init__()
        dense = []
        inter_channels = original_channels
        for i in range(num_dense):
            dense.append(dense_layer(inter_channels, original_channels))
            inter_channels += original_channels
        self.dense_block = nn.Sequential(*dense)
        self.fusion = nn.Conv2d(inter_channels, original_channels,
                                kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        output = self.dense_block(x)
        output = self.fusion(output)
        return x + output
