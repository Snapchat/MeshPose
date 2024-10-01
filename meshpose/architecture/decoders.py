import torch

MESH_CENTER_INDICES = [81, 82, 85, 86, 91, 92, 93, 94, 109, 111, 113, 217, 220, 237, 239, 283]
MESH_CENTER_WEIGHTS = [0.009604673832654953, 0.03351994603872299, 0.19301486015319824, 0.045285675674676895,
                       0.003511656541377306, 0.012976124882698059, 0.002114641945809126, 0.19195371866226196,
                       0.019679106771945953, 0.0010375987039878964, 0.13418574631214142, 0.00015269737923517823,
                       0.22274497151374817, 0.01192322839051485, 0.0974457710981369, 0.02076236717402935]
VERTEX_SEAMS = [[4, 87], [59, 368], [60, 369], [61, 370], [62, 371], [63, 372], [64, 373], [65, 374], [71, 378],
                [76, 379], [174, 431], [175, 432], [176, 433], [177, 434], [178, 441], [185, 446], [300, 500],
                [301, 501], [302, 502], [303, 503], [304, 510], [309, 515]]


class MeshUpsamplingDecoder(torch.nn.Module):
    def __init__(self, checkpoint, device, num_vert_low_poly=518, num_vert_upsampler_intermediate=1723,
                 num_vert_high_poly=6890):
        super().__init__()
        self.upsampling = torch.nn.Linear(num_vert_low_poly, num_vert_upsampler_intermediate).to(device)
        self.upsampling2 = torch.nn.Linear(num_vert_upsampler_intermediate, num_vert_high_poly).to(device)
        data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        data_dict = data['model']['VERTEX_UPSAMPLE']
        self.load_state_dict(data_dict, strict=True)

        self.center_indices = MESH_CENTER_INDICES
        self.center_weights = torch.tensor(MESH_CENTER_WEIGHTS).view(1, -1, 1).to(device)

    def get_center_of_mesh(self, vertices):
        return (vertices[:, self.center_indices, :] * self.center_weights).sum(1).unsqueeze(0)

    def forward(self, mesh):
        with torch.no_grad():
            center = self.get_center_of_mesh(mesh)
            mesh = mesh - center
            mesh = self.upsampling(mesh.permute(0, 2, 1))
            mesh = self.upsampling2(mesh).permute(0, 2, 1)
            mesh = mesh + center
        return mesh


class MeshPoseBranchFusion:
    def __init__(self, scale_factor=4, conv1d_out_bins=64, regression_height_multiplier=3,
                 regression_width_multiplier=4, regression_depth_multiplier=4):
        self.scale_factor = scale_factor
        self.seams = VERTEX_SEAMS
        self.conv1d_out_bins = conv1d_out_bins
        self.regression_height_multiplier = regression_height_multiplier
        self.regression_width_multiplier = regression_width_multiplier
        self.regression_depth_multiplier = regression_depth_multiplier

    def __call__(self, output):

        # X, Y coordinates as computed from the vertexpose head
        vertexpose_coordinates = output['SP_MESHPOSE']['coordinates']

        # X, Y, Z coordinates as computed from the regression head
        _, _, vw, vh = output['SP_MESHPOSE']['heatmap'].shape
        regressed_vertices = torch.cat([
            self.regression_height_multiplier * vw / self.conv1d_out_bins * output['SP_VERTEX_X'] - vw,
            self.regression_width_multiplier * vh / self.conv1d_out_bins * output['SP_VERTEX_Y'] - vh,
            self.regression_depth_multiplier * vh / self.conv1d_out_bins * output['SP_VERTEX_Z'] - 2 * vh],
            2)

        # Visibility values used for output combination
        vertex_w = torch.sigmoid(output['SP_VISIBILITY'])

        # Combined vertex locations
        combined_vertices = regressed_vertices.detach().clone()

        combined_vertices[:, :, 0] = (vertexpose_coordinates[:, :, 0] * vertex_w +
                                      regressed_vertices[:, :, 0].clone() * (1 - vertex_w))

        combined_vertices[:, :, 1] = (vertexpose_coordinates[:, :, 1] * vertex_w +
                                      regressed_vertices[:, :, 1].clone() * (1 - vertex_w))

        for v0, v1 in self.seams:
            merged_verts = 0.5 * (combined_vertices[:, v0, :] + combined_vertices[:, v1, :])
            combined_vertices[:, v0, :] = merged_verts
            combined_vertices[:, v1, :] = merged_verts

        return combined_vertices * self.scale_factor, vertex_w
