import torch
import torch.nn as nn


class SoftArgmax2D(nn.Module):
    def __init__(self, gain_softmax=10, output_height=96, output_width=72):
        super().__init__()
        self.gain_softmax = gain_softmax
        xx, yy = torch.meshgrid(torch.arange(output_height), torch.arange(output_width), indexing='ij')
        self.grid = torch.nn.Parameter(torch.stack((xx, yy), dim=2), requires_grad=False)

    def soft_argmax(self, heatmap):
        batch_size, dim_channels, height, width = heatmap.shape
        softmax_output = nn.functional.softmax(heatmap.view(batch_size, dim_channels, height * width) *
                                               self.gain_softmax, dim=2)
        softmax_output = softmax_output.view(batch_size, dim_channels, height, width, 1)
        conv = softmax_output * self.grid.view(1, 1, height, width, 2)
        coords = conv.sum(dim=2).sum(dim=2)
        coords = coords[:, :, [1, 0]]
        return coords

    def forward(self, features):
        coords = self.soft_argmax(features)
        return {'coordinates': coords, 'heatmap': features}


class SoftArgmax1D(nn.Module):
    def __init__(self, heatmap_size, alpha=1.0):
        super().__init__()
        self.register_buffer("positions", torch.arange(heatmap_size).float())
        self.alpha = alpha

    def forward(self, heatmap1d):
        heatmap1d = nn.functional.softmax(self.alpha * heatmap1d, 2)
        coordinate_expectation = (heatmap1d * self.positions).sum(dim=2, keepdim=True)
        return coordinate_expectation


class Conv1DNet(nn.Module):
    def __init__(self, joint_num=518, in_channels=256, hiddens=256, out_bins=64, output_range=96):
        super().__init__()
        self.in_channels = in_channels
        self.output_range = output_range
        self.out_bins = out_bins
        self.hiddens = hiddens
        self.joint_num = joint_num
        self.conv_z_1 = self.make_conv1d_layers([self.in_channels, self.hiddens * self.out_bins])
        self.conv_z_2 = self.make_conv1d_layers([self.hiddens, self.joint_num], bnrelu_final=False)

    @staticmethod
    def make_conv1d_layers(feat_dims, kernel=1, stride=1, padding=0, bnrelu_final=True):
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i + 1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                ))
            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def conv1d_net(self, img_feat):
        img_feat_z = img_feat.mean((2, 3))[:, :, None]  # b x c x h x w -> b x c
        img_feat_z = self.conv_z_1(img_feat_z)  # b x c -> b x hiddens*out_bins
        # b x hiddens*out_bins ->  b x c -> b x hiddens x out_bins
        img_feat_z = img_feat_z.view(-1, self.hiddens, self.out_bins)
        return self.conv_z_2(img_feat_z)  # b x hiddens x out_bins -> b x joints_num x out_bins


class DepthNet(Conv1DNet):
    def __init__(self, joint_num=518, in_channels=256, hiddens=256, out_bins=64, softmax_alpha=1.0):
        super().__init__(joint_num=joint_num, in_channels=in_channels, hiddens=hiddens, out_bins=out_bins)
        self.softargmax1D = SoftArgmax1D(self.out_bins, softmax_alpha)

    def forward(self, img_feat):

        heatmap_z = self.conv1d_net(img_feat)
        coord_z = self.softargmax1D(heatmap_z)  # b x joints_num x out_bins -> b x joints_num
        coord_z = coord_z * self.output_range / self.out_bins  # [0, self.out_bins] range to [0, self.output_range]

        return coord_z


class VisNet(Conv1DNet):
    def __init__(self, joint_num=518, in_channels=256, hiddens=256, out_bins=64):
        super().__init__(joint_num=joint_num, in_channels=in_channels, hiddens=hiddens, out_bins=out_bins)

    def forward(self, img_feat):

        heatmap_z = self.conv1d_net(img_feat)
        vis = heatmap_z.mean(-1)  # b x hiddens x out_bins -> b x joints_num
        return vis


class EncDecModel(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else None
        self.decoder = decoder(encoder=self.encoder) if decoder is not None else None

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        if self.decoder is not None:
            x = self.decoder(x)
        return x


class SequentialModel(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.model = nn.ModuleList(modules)

    def forward(self, x):
        for module_ in self.model:
            x = module_(x)
        return x


class Model(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class MultiBranchNet(nn.Module):
    def __init__(self, filters_last=256):
        super().__init__()

        branches = [
            {'channels': 518, 'branch_name': 'SP_MESHPOSE', 'model': SoftArgmax2D()},
            {'channels': None, 'branch_name': 'SP_VERTEX_X', 'model': DepthNet()},
            {'channels': None, 'branch_name': 'SP_VERTEX_Y', 'model': DepthNet()},
            {'channels': None, 'branch_name': 'SP_VERTEX_Z', 'model': DepthNet()},
            {'channels': None, 'branch_name': 'SP_VISIBILITY', 'model': VisNet()}]

        def channel_adapter(in_channels, out_channels):
            last_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            return last_conv

        self.head_layers = nn.ModuleDict()
        self.branch_models = nn.ModuleDict()

        for spec in branches:
            branch = spec["branch_name"]

            if spec['channels'] is None:
                self.head_layers[branch] = spec['model']
            else:
                self.head_layers[branch] = channel_adapter(in_channels=filters_last, out_channels=spec["channels"])
                self.branch_models[branch] = spec['model']

    def forward(self, features):
        branch_to_outputs = {}
        for branch, head_layer in self.head_layers.items():
            branch_to_outputs[branch] = head_layer(features)

        for branch, model in self.branch_models.items():
            branch_to_outputs[branch] = model(branch_to_outputs[branch])

        return branch_to_outputs
