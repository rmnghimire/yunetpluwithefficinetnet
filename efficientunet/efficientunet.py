from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]


        self.encoder = encoder
        self.concat_input = concat_input
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        self.doubleconv01 = double_conv(40,nb_filter[0])
        self.doubleconv11 = double_conv(64, nb_filter[1])
        self.doubleconv21 = double_conv(120, nb_filter[2])

        self.doubleconv02 = double_conv(112, nb_filter[0])
        self.doubleconv12 = double_conv(216, nb_filter[1])

        self.doubleconv03 = double_conv(144, nb_filter[0])

        self.doubleconv22 = double_conv(424, nb_filter[2])
        self.doubleconv13 = double_conv(280, nb_filter[1])
        self.doubleconv04 = double_conv(176, nb_filter[0])


        self.doubleconve31 = up_conv(512,32)
        self.doubleconve22 = up_conv(128, 32)
        self.doubleconve13 = up_conv(64, 32)

        self.similar = up_conv(32,32)
        self.final =  up_conv(32,2)
        if self.concat_input:
            self.up_conv_input = up_conv(32,16)
            self.double_conv_input = double_conv(19,32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        nb_filter = [32, 64, 128, 256, 512]

        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x4_0 = blocks.popitem()
        x3_0 = blocks.popitem()[1]
        x2_0 = blocks.popitem()[1]
        x1_0 = blocks.popitem()[1]
        x0_0 = blocks.popitem()[1]




        x4_0 = self.up_conv1(x4_0)
        x3_1 = torch.cat([x4_0, x3_0], dim=1)
        x3_1 = self.double_conv1(x3_1)


        # x3_1 = self.up_conv2(x3_1)
        x30_upsample = self.up(x3_0)
        x2_1 = torch.cat([x2_0, x30_upsample], dim=1)
        x2_1 = self.doubleconv21(x2_1)
        x2_2 = torch.cat([(self.up_conv2(x3_1)), x2_0,x2_1], dim=1)
        x2_2 = self.doubleconv22(x2_2)
        # x2_2 = self.double_conv2(x2_2)

        # print(blocks.popitem()[1].shape, "3")

        # x2_2 = self.up(x2_2)
        x20_upsample = self.up(x2_0)
        x1_1 = torch.cat([x1_0, x20_upsample], dim=1)
        x1_1 = self.doubleconv11(x1_1)

        x21_upsample = self.up(x2_1)
        x1_2 = torch.cat([x1_0, x1_1, x21_upsample], dim=1)
        x1_2 = self.doubleconv12(x1_2)
        x1_3 = torch.cat([self.up(x2_2), x1_0,x1_1,x1_2], dim=1)

        x1_3 = self.doubleconv13(x1_3)
        # print(blocks.popitem()[1].shape, "4")

        # x1_3 = self.up(x1_3)
        x10_upsample = self.up(x1_0)
        x0_1 = torch.cat([x0_0, x10_upsample], dim=1)


        x0_1 = self.doubleconv01(x0_1)
        x11_upsample = self.up(x1_1)
        x0_2 = torch.cat([x0_1, x0_0, x11_upsample], dim=1)
        x0_2 = self.doubleconv02(x0_2)
        x12_upsample = self.up(x1_2)
        x0_3 = torch.cat([x0_0, x0_1, x0_2, x12_upsample], dim=1)
        x0_3 = self.doubleconv03(x0_3)
        x0_4 = torch.cat([self.up(x1_3), x0_0,x0_1,x0_2,x0_3], dim=1)
        x0_4 = self.doubleconv04(x0_4)


        # print(x4_0.shape, "x40")
        # print(x3_1.shape, "31")
        # print(x2_2.shape, "x22")
        # print(x1_3.shape, "x13")
        # print(x0_4.shape, "x04")

        t1 = self.similar(self.doubleconve31(x3_1))
        t1 = self.similar(t1)
        t1 = self.final(t1)

        t2 = self.similar(self.doubleconve22(x2_2))
        t2 = self.final(t2)
        t3 = self.doubleconve13(x1_3)
        t3 = self.final(t3)

        t4 = self.final(x0_4)
        print(t1.shape, "t1")
        print(t2.shape, "t2")
        print(t3.shape, "t3")
        print(t4.shape, "t4")

        if self.concat_input:
            x=((t1+t2+t3+t4)/4)




        print(x.shape, "final sakkiyeko x")
        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
