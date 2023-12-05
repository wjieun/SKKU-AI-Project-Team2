import torch
import torch.nn as nn
import torchvision

# Model

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.double_conv = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
              nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
          )

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.double_conv(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 5

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=True)
        down_blocks = []
        up_blocks = []

        # 입력 채널을 n_channels로 조정
        '''
        self.input_block = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.vgg.features[:6]
        )
        '''

        self.block1 = nn.Sequential(self.vgg.features[:6])
        self.block2 = nn.Sequential(self.vgg.features[6:13])
        self.block3 = nn.Sequential(self.vgg.features[13:20])
        self.block4 = nn.Sequential(self.vgg.features[20:27])
        self.block5 = nn.Sequential(self.vgg.features[27:34])

        self.bottleneck = nn.Sequential(self.vgg.features[34:])

        self.bridge = double_conv(512, 1024)

        self.upblock6 = UpBlockForUNetWithResNet50(1024, 512)
        self.upblock7 = UpBlockForUNetWithResNet50(256 + 512, 256, 512, 256)
        self.upblock8 = UpBlockForUNetWithResNet50(128 + 256, 128, 256, 128)
        self.upblock9 = UpBlockForUNetWithResNet50(64 + 128, 64, 128, 64)
        self.upblock10 = UpBlockForUNetWithResNet50(32 + 64, 32, 64, 32)

        self.out = nn.Conv2d(32, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)


        bottleneck = self.bottleneck(block5)
        x = self.bridge(bottleneck)

        x = self.upblock6(x, block5)
        x = self.upblock7(x, block4)
        x = self.upblock8(x, block3)
        x = self.upblock9(x, block2)
        x = self.upblock10(x, block1)


        output_feature_map = x
        x = self.out(x)
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
        

# Use as below
# device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
# model = UNetWithResnet50Encoder(n_channels=3, n_classes=1).to(device)
# model.load_state_dict(torch.load('/content/drive/MyDrive/인지프/checkpoints/best_model_vgg16.pth'))
# model.eval()