import torch.nn as nn


HRNET_32_256_FILT = {'STAGE1': {'BLOCK': 'BOTTLENECK',
                                'NUM_BLOCKS': [4],
                                'NUM_CHANNELS': [64]},
                     'STAGE2': {'BLOCK': 'BASIC',
                                'NUM_BLOCKS': [4, 4],
                                'NUM_BRANCHES': 2,
                                'NUM_CHANNELS': [32, 64],
                                'NUM_HIDDENS': [[0, 32], [64, 0]],
                                'NUM_MODULES': 1},
                     'STAGE3': {'BLOCK': 'BASIC',
                                'NUM_BLOCKS': [4, 4, 4],
                                'NUM_BRANCHES': 3,
                                'NUM_CHANNELS': [32, 64, 128],
                                'NUM_HIDDENS': [[0, 32, 32], [64, 0, 64], [128, 128, 0]],
                                'NUM_MODULES': 4},
                     'STAGE4': {'BLOCK': 'BASIC',
                                'NUM_BLOCKS': [4, 4, 4, 4],
                                'NUM_BRANCHES': 4,
                                'NUM_CHANNELS': [32, 64, 128, 256],
                                'NUM_HIDDENS': [[256, 256, 256, 256]],
                                'NUM_MODULES': 3}}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_bn_pairs = [("conv1", "bn1"), ("conv2", "bn2")]

    @classmethod
    def get_expansion(cls):
        return 1

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return self.relu2(out + residual)


class Bottleneck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None
    ):
        super().__init__()
        self.expansion = Bottleneck.get_expansion()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.conv_bn_pairs = [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3"),]

    @classmethod
    def get_expansion(cls):
        return 4

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return self.relu3(out + residual)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        num_hiddens,
        multi_scale_output=True
    ):
        super().__init__()

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        def _relu_fn():
            return nn.ReLU(inplace=True)

        self.relu_fn = _relu_fn

        def add_fn(x, y):
            return x + y

        self.fuse_same_fn = add_fn
        self.fuse_same_act_fn = nn.Identity

        self.fuse_other_fn = add_fn
        self.fuse_other_act_fn = nn.Identity

        if num_hiddens is None:
            num_hiddens = [None] * (num_branches if self.multi_scale_output else 1)
            for i in range(num_branches if self.multi_scale_output else 1):
                num_hiddens[i] = [None] * num_branches
                for j in range(num_branches):
                    if j != i:
                        num_hiddens[i][j] = self.num_inchannels[i]

        self.num_outchannels = [None] * (num_branches if self.multi_scale_output else 1)
        for i in range(num_branches if self.multi_scale_output else 1):
            self.num_outchannels[i] = [h or c for h, c in zip(num_hiddens[i], self.num_inchannels)]
            self.num_outchannels[i] = self.num_outchannels[i][0]
        self.num_hiddens = num_hiddens
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        (self.fuse_layers, self.fuse_relus, self.fuse_activations) = self._make_fuse_layers()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = nn.Identity()
        expansion = block.get_expansion()

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        num_hiddens = self.num_hiddens
        fuse_layers = []
        fuse_relus = []
        fuse_activations = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i or (j == i and num_hiddens[i][j]):
                    upsample_modules = []
                    num_inchannels_conv = num_inchannels[j]
                    if num_hiddens[i][j]:
                        num_outchannels_conv = num_hiddens[i][j]
                        upsample_modules.extend(
                            [
                                conv1x1(num_inchannels_conv, num_outchannels_conv),
                                nn.BatchNorm2d(num_outchannels_conv),
                            ]
                        )

                    if j > i:
                        upsample_modules.append(
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode='bilinear',
                                align_corners=True
                            )
                        )
                    fuse_layer.append(nn.Sequential(*upsample_modules))
                elif j == i and not num_hiddens[i][j]:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv = num_hiddens[i][j]
                            conv3x3s_modules = [
                                conv3x3(
                                    num_inchannels[j],
                                    num_outchannels_conv,
                                    2,
                                ),
                                nn.BatchNorm2d(num_outchannels_conv),
                            ]
                            conv3x3s.append(nn.Sequential(*conv3x3s_modules))
                        else:
                            num_outchannels_conv = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    conv3x3(
                                        num_inchannels[j],
                                        num_outchannels_conv,
                                        2,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv),
                                    self.relu_fn(),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
            fuse_relus.append(self.relu_fn())

            fuse_activation = [nn.Identity()]
            for j in range(1, num_branches):
                if i == j:
                    fuse_activation.append(self.fuse_same_act_fn())
                else:
                    fuse_activation.append(self.fuse_other_act_fn())
            fuse_activations.append(nn.ModuleList(fuse_activation))

        return nn.ModuleList(fuse_layers), nn.ModuleList(fuse_relus), nn.ModuleList(fuse_activations)

    def get_num_outchannels(self):
        return self.num_outchannels

    def forward(self, x):

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = self.fuse_same_fn(y, x[j])
                else:
                    y = self.fuse_other_fn(y, self.fuse_layers[i][j](x[j]))
                y = self.fuse_activations[i][j](y)
            x_fuse.append(self.fuse_relus[i](y))

        return x_fuse


class Hrnet32(nn.Module):
    def __init__(self, hrnet_model_config=None):
        super().__init__()

        if hrnet_model_config is None:
            hrnet_model_config = HRNET_32_256_FILT

        self.inplanes = 64
        self.blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}

        def _get_relu_fn():
            return nn.ReLU(inplace=True)

        self.relu_fn = _get_relu_fn
        self.relu1 = self.relu_fn()
        self.relu2 = self.relu_fn()
        self.conv_bn_pairs = []

        # STAGE 1
        # stem net
        in_channels = 3
        self.conv1 = conv3x3(in_channels, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)

        self.stage1_cfg = hrnet_model_config["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = self.blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, num_channels, num_blocks)
        stage1_out_channel = block.get_expansion() * num_channels

        self.conv_bn_pairs += [("conv1", "bn1"), ("conv2", "bn2")]

        # STAGE 2
        self.stage2, self.stage2_cfg, self.transition1, pre_stage_channels = self.create_stage(
            hrnet_model_config, "STAGE2", [stage1_out_channel]
        )

        # STAGE 3
        self.stage3, self.stage3_cfg, self.transition2, pre_stage_channels = self.create_stage(
            hrnet_model_config, "STAGE3", pre_stage_channels
        )

        # STAGE 4
        self.stage4, self.stage4_cfg, self.transition3, pre_stage_channels = self.create_stage(
            hrnet_model_config, "STAGE4", pre_stage_channels, multi_scale_output=False
        )

        self.filters_last = pre_stage_channels[0]

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            conv3x3(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                1,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            self.relu_fn(),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            conv3x3(inchannels, outchannels, 2),
                            nn.BatchNorm2d(outchannels),
                            self.relu_fn(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Identity()
        expansion = block.get_expansion()
        if stride != 1 or self.inplanes != planes * expansion:
            downsample_modules = [
                nn.Conv2d(self.inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            ]
            downsample = nn.Sequential(*downsample_modules)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        num_hiddens = layer_config["NUM_HIDDENS"]
        block = self.blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
                reset_num_hiddens = num_hiddens
            else:
                reset_multi_scale_output = True
                reset_num_hiddens = None

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_num_hiddens,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_outchannels()

        return nn.Sequential(*modules), num_inchannels

    @staticmethod
    def stage_forward(in_list, num_branches, transition, stage):
        x_list = []
        for i in range(num_branches):
            if not isinstance(transition[i], nn.Identity):
                x_list.append(transition[i](in_list[-1]))
            else:
                x_list.append(in_list[i])
        return stage(x_list)

    @staticmethod
    def stage1_forward(x, num_branches, transition, stage):
        x_list = []
        for i in range(num_branches):
            x_list.append(transition[i](x))

        return stage(x_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.layer1(x)

        x_list = self.stage1_forward(x, self.stage2_cfg["NUM_BRANCHES"], self.transition1, self.stage2)
        x_list = self.stage_forward(x_list, self.stage3_cfg["NUM_BRANCHES"], self.transition2, self.stage3)
        x_list = self.stage_forward(x_list, self.stage4_cfg["NUM_BRANCHES"], self.transition3, self.stage4)

        return x_list[0]

    def create_stage(self, hrnet_config, stage_num, prev_stage_out_channels, multi_scale_output=True):
        stage_cfg = hrnet_config[stage_num]
        block = self.blocks_dict[stage_cfg["BLOCK"]]
        expansion = block.get_expansion()
        num_channels = stage_cfg["NUM_CHANNELS"]
        num_channels = [c * expansion for c in num_channels]
        transition_layer = self._make_transition_layer(prev_stage_out_channels, num_channels)
        stage, pre_stage_channels = self._make_stage(stage_cfg, num_channels, multi_scale_output)
        return stage, stage_cfg, transition_layer, pre_stage_channels
