import torch
from torch import nn

from modules import DepthWiseDilatedResidualBlock, DepthGuidedAttentionBlock, ResidualDenseBlock


class DehazingNet(nn.Module):
    def __init__(self, num_features=64):
        super(DehazingNet, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=num_features, num_channels=num_features),
            nn.SELU(inplace=True)
        )
        self.body = nn.Sequential(
            DepthWiseDilatedResidualBlock(num_features, 1),
            DepthWiseDilatedResidualBlock(num_features, 1),
            DepthWiseDilatedResidualBlock(num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, 4),
            DepthWiseDilatedResidualBlock(num_features, 8),
            DepthWiseDilatedResidualBlock(num_features, 4),
            DepthWiseDilatedResidualBlock(num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, 1),
            DepthWiseDilatedResidualBlock(num_features, 1)
        )

        self.attention = DepthGuidedAttentionBlock(num_features, 64, 128)

        self.tail = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, depth):
        x = (x - self.mean) / self.std
        f = self.head(x)
        f = self.body(f)
        f = self.attention(f, depth, 8)
        r = self.tail(f)

        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        return x


class MultiScaleResidualDenseNet(nn.Module):
    def __init__(self, feature_channels=32, num_dense=5):
        super(MultiScaleResidualDenseNet, self).__init__()

        ############## selu initialization ##############

        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############## depth prediction ##############

        # MRDB-1 [32,1024,512]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.Conv2d(3, feature_channels, kernel_size=1),
            nn.GroupNorm(num_groups=feature_channels, num_channels=feature_channels),
            nn.SELU(inplace=True)
        )
        self.rdb1_1 = ResidualDenseBlock(feature_channels, num_dense)
        self.rdb1_2 = ResidualDenseBlock(feature_channels, num_dense)
        self.concat1 = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1, padding=0, stride=1)
        )

        # MRDB-2 [32,512,256]
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=4, stride=2, padding=1, groups=feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(num_groups=feature_channels, num_channels=feature_channels),
            nn.SELU(inplace=True)
        )
        self.rdb2_1 = ResidualDenseBlock(feature_channels, num_dense)
        self.rdb2_2 = ResidualDenseBlock(feature_channels, num_dense)
        self.concat2 = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1, padding=0, stride=1)
        )

        # MRDB-3 [32,256,128]
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=4, stride=2, padding=1, groups=feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(num_groups=feature_channels, num_channels=feature_channels),
            nn.SELU(inplace=True)
        )
        self.rdb3_1 = ResidualDenseBlock(feature_channels, num_dense)
        self.rdb3_2 = ResidualDenseBlock(feature_channels, num_dense)
        self.concat3 = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1, padding=0, stride=1)
        )

        # deconv1 [32,256,128] -> [32,512,256]
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=feature_channels, num_channels=feature_channels),
            nn.SELU(inplace=True)
        )

        # deconv2 [32,512,256] -> [32,1024,512]
        self.deconv2 = nn.Sequential(
            # concat
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1, padding=0, stride=1),

            nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=feature_channels, num_channels=feature_channels),
            nn.SELU(inplace=True)
        )
        # depth_pred
        self.depth_pred = nn.Sequential(
            # concat
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1, padding=0, stride=1),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        ############## selu initialization ##############

        x = (x - self.mean) / self.std

        ############## depth prediction ##############

        # MRDB-1
        f1 = self.conv1(x)
        f_rd1_1 = self.rdb1_1(f1)
        f_rd1_2 = self.rdb1_2(f_rd1_1)
        f_cat1 = torch.cat((f_rd1_1, f_rd1_2), 1)
        f_out1 = self.concat1(f_cat1)

        # MRDB-2
        f2 = self.conv2(f_out1)
        f_rd2_1 = self.rdb2_1(f2)
        f_rd2_2 = self.rdb2_2(f_rd2_1)
        f_cat2 = torch.cat((f_rd2_1, f_rd2_2), 1)
        f_out2 = self.concat2(f_cat2)

        # MRDB-3
        f3 = self.conv3(f_out2)
        f_rd3_1 = self.rdb3_1(f3)
        f_rd3_2 = self.rdb3_2(f_rd3_1)
        f_cat3 = torch.cat((f_rd3_1, f_rd3_2), 1)
        f_out3 = self.concat3(f_cat3)

        # deconv1
        f4 = self.deconv1(f_out3)

        # deconv2
        f5 = torch.cat((f4, f_out2), 1)
        f5 = self.deconv2(f5)

        # depth_pred
        f6 = torch.cat((f5, f_out1), 1)
        depth_pred = self.depth_pred(f6)

        return depth_pred


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dehaze = DehazingNet()
        self.depth = MultiScaleResidualDenseNet()

    def forward(self, x):
        dep = self.depth(x)
        res = self.dehaze(x, dep)
        return res, dep
