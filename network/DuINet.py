import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)

# spatial attention
class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# channel attention
class CAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class DuINet(nn.Module):
    def __init__(self, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DuINet, self).__init__()
        kernel_size = 3
        padding = 1

        # DnCNN
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )

        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),)
        # 成像
        self.conv1_7 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=1, padding=0,bias=False)


        # u-net编码部分
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True),
        )
        self.encoder1 = nn.Sequential(Conv_Block(128,64),nn.MaxPool2d(kernel_size=2, stride=2),)
        self.encoder2 = nn.Sequential(Conv_Block(128,128),nn.MaxPool2d(kernel_size=2, stride=2),)
        self.encoder3 = nn.Sequential(Conv_Block(128,256),nn.MaxPool2d(kernel_size=2, stride=2),)

        self.conv = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=kernel_size,padding=padding,bias=False)

       # u-net解码部分
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=padding,bias=False),
            nn.BatchNorm2d(256, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(512, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=padding,bias=False),
            nn.BatchNorm2d(256, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(256, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding,bias=False),
            nn.BatchNorm2d(128, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(128, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(128, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(64, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True), )

        self.conv2_3 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=1, padding=0,bias=False)


        # skip-connection
        self.skip1 = nn.MaxPool2d(2,stride=2)
        self.skip2 = nn.PixelShuffle(2)
        self.skip3 = nn.MaxPool2d(5,stride=8)
        self.skip4 = nn.Conv2d(in_channels=256+image_channels,out_channels=64,kernel_size = 1,padding = 0,bias=False)
        self.spatial_attention = SAM()
        self.channel_attention = CAM(channel=256+image_channels)

        # 重构
        self.reconstruct = nn.Conv2d(in_channels=image_channels*2, out_channels=image_channels, kernel_size=1, stride=1, padding=0)

        # self.weight = self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if 0 <= m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif -clip_b < m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self,x):
        y = x
        # high-resolution1
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x1_1)
        x1_3 = self.conv1_3(x1_2)

        # encoder
        x2_1 = self.conv2_1(x)
        x2_2 = self.encoder1(torch.cat((x2_1, x1_1), dim=1))
        x2_3 = self.encoder2(torch.cat((x2_2, self.skip1(x1_2)), dim=1))
        x2_4 = self.encoder3(x2_3)
        x2_5 = self.conv(x2_4)

        x1_3_1 = self.skip3(self.spatial_attention(torch.cat((x1_3, x), dim=1)))
        x2_5_1 = self.skip4(self.channel_attention(torch.cat((x2_5, self.skip3(x)), dim=1)))
        x1_4 = self.conv1_4(x1_3 * x2_5_1)

        # decoder
        x2_6 = self.decoder1(x2_5 * x1_3_1)
        x2_7 = self.decoder2(torch.cat((x2_6, x2_3), dim=1))
        x2_8 = self.decoder3(torch.cat((x2_7, x2_2), dim=1))
        x2_9 = self.conv2_2(torch.cat((x2_8, x2_1), dim=1))

        # high-resolution2
        x1_5 = self.conv1_5(torch.cat((x1_4,self.skip2(x2_7)),dim=1))
        x1_6 = self.conv1_6(torch.cat((x1_5,x2_8),dim=1))
        x1_7 = self.conv1_7(x1_6)
        x1 = y - x1_7

        x2_10 = self.conv2_3(x2_9)
        x2 = y - x2_10

        out = torch.cat((x1,x2),dim=1)
        out = self.reconstruct(out)

        return y-out    # residual


