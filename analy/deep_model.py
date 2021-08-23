import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2_Emotion(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        super(network_29layers_v2_Emotion, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)
        self.fcOut = nn.Linear(256, num_classes, bias=True)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        out = self.fcOut(x)
        return out

def EmotionNet(**kwargs):
    model = network_29layers_v2_Emotion(resblock, [1, 2, 3, 4], **kwargs)
    return model



class HeadposeNet(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(HeadposeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll




def padding_same_conv2d(input_size, in_c, out_c, kernel_size=4, stride=1):
    output_size = input_size // stride
    padding_num = stride * (output_size - 1) - input_size + kernel_size
    if padding_num % 2 == 0:
        return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding_num // 2, bias=False))
    else:
        return nn.Sequential(
            nn.ConstantPad2d((padding_num // 2, padding_num // 2 + 1, padding_num // 2, padding_num // 2 + 1), 0),
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        )

class resBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=1, input_size=None):
        super().__init__()
        assert kernel_size == 4
        self.shortcut = lambda x: x
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

        main_layers = [
            nn.Conv2d(in_c, out_c // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
        ]

        main_layers.extend([
            *padding_same_conv2d(input_size, out_c // 2, out_c // 2, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)])

        main_layers.extend(
            padding_same_conv2d(input_size, out_c // 2, out_c, kernel_size=1, stride=1)
        )
        self.main = nn.Sequential(*main_layers)
        self.activate = nn.Sequential(
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        shortcut_x = self.shortcut(x)
        main_x = self.main(x)
        x = self.activate(shortcut_x + main_x)
        return x


class upBlock(nn.Module):
    def __init__(self, in_c, out_c, conv_num=2):
        super().__init__()
        additional_conv = []
        layer_length = 4

        for i in range(1, conv_num+1):
            additional_conv += [
                nn.ConstantPad2d((2, 1, 2, 1), 0),
                nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
                nn.ReLU(inplace=True)
            ]
        self.main = nn.Sequential(
            # nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
            *additional_conv
            )

    def forward(self, x):
        x = self.main(x)
        return x


class LandmarkNet(nn.Module):
    def __init__(self, in_channel, b_sz):
        super().__init__()
        size = 16
        self.input_conv = nn.Sequential( #*[
            *padding_same_conv2d(256, in_channel, size, kernel_size=4, stride=1),  # 256x256x16
            nn.BatchNorm2d(size, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
            # ]
        )
        self.down_conv_1 = resBlock(size, size * 2, kernel_size=4, stride=2, input_size=256)  # 128x128x32
        self.down_conv_2 = resBlock(size * 2, size * 2, kernel_size=4, stride=1, input_size=128)  # 128x128x32
        self.down_conv_3 = resBlock(size * 2, size * 4, kernel_size=4, stride=2, input_size=128)  # 64x64x64
        self.down_conv_4 = resBlock(size * 4, size * 4, kernel_size=4, stride=1, input_size=64)  # 64x64x64
        self.down_conv_5 = resBlock(size * 4, size * 8, kernel_size=4, stride=2, input_size=64)  # 32x32x128
        self.down_conv_6 = resBlock(size * 8, size * 8, kernel_size=4, stride=1, input_size=32)  # 32x32x128
        self.down_conv_7 = resBlock(size * 8, size * 16, kernel_size=4, stride=2, input_size=32)  # 16x16x256
        self.down_conv_8 = resBlock(size * 16, size * 16, kernel_size=4, stride=1, input_size=16)  # 16x16x256
        self.down_conv_9 = resBlock(size * 16, size * 32, kernel_size=4, stride=2, input_size=16)  # 8x8x512
        self.down_conv_10 = resBlock(size * 32, size * 32, kernel_size=4, stride=1, input_size=8)  # 8x8x512

        self.center_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size * 32, size * 32, kernel_size=4, stride=1, padding=3, bias=False),  # 8x8x512
            nn.BatchNorm2d(size * 32, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )

        self.up_conv_5 = upBlock(size * 32, size * 16)  # 16x16x256
        self.up_conv_4 = upBlock(size * 16, size * 8)  # 32x32x128
        self.up_conv_3 = upBlock(size * 8, size * 4)  # 64x64x64

        self.up_conv_2 = upBlock(size * 4, size * 2, 1)  # 128x128x32
        self.up_conv_1 = upBlock(size * 2, size, 1)  # 256x256x16

        self.output_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.Sigmoid()
        )
        self.idx_x = torch.tensor([15,  22,  26,  32,  45,  67,  91, 112, 128, 143, 164, 188, 210,
        223, 229, 233, 240,  58,  71,  85,  97, 106, 149, 158, 170, 184,
        197, 128, 128, 128, 128, 117, 122, 128, 133, 138,  78,  86,  95,
        102,  96,  87, 153, 160, 169, 177, 168, 159, 108, 116, 124, 128,
        131, 139, 146, 137, 132, 128, 123, 118, 110, 122, 128, 133, 145,
        132, 128, 123]).long()
        self.idx_y = torch.tensor([96, 118, 141, 165, 183, 190, 188, 187, 193, 187, 188, 190, 183,
        165, 141, 118,  96,  49,  42,  39,  40,  42,  42,  40,  39,  42,
         49,  59,  73,  86,  96, 111, 113, 115, 113, 111,  67,  60,  61,
         65,  68,  69,  65,  61,  60,  67,  69,  68, 142, 131, 127, 128,
        127, 131, 142, 148, 150, 150, 150, 148, 141, 135, 134, 135, 142,
        143, 142, 143]).long()


    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_conv_1(x)
        x = self.down_conv_2(x)
        x = self.down_conv_3(x)
        x = self.down_conv_4(x)
        x = self.down_conv_5(x)
        x = self.down_conv_6(x)
        x = self.down_conv_7(x)
        x = self.down_conv_8(x)
        x = self.down_conv_9(x)
        x = self.down_conv_10(x)

        x = self.center_conv(x)

        x = self.up_conv_5(x)
        x = self.up_conv_4(x)
        x = self.up_conv_3(x)
        x = self.up_conv_2(x)
        x = self.up_conv_1(x)
        x = self.output_conv(x)

        x = x.permute(0,2, 3, 1)
        x = x[:, self.idx_y, self.idx_x, 0:2]
        x = x.view(x.size(0), 68*2)

        return x