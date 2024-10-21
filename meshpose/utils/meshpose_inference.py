import torch

from meshpose.preprocessing import ImagePreprocessing
from meshpose.architecture.model import MeshPoseModel
from meshpose.utils import affine_tranform_3d
from meshpose.architecture.decoders import MeshUpsamplingDecoder, MeshPoseBranchFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MeshPoseInference:
    def __init__(self,
                 main_checkpoint="checkpoints/hrnet32_checkpoint_EXTERNAL.pth.tar",
                 upsampler_checkpoint="checkpoints/upsampler_checkpoint_EXTERNAL.pth.tar",
                 scale_bbox=1.1):
        self.img_preprocessing = ImagePreprocessing(scale_bbox=scale_bbox)
        self.meshpose_model = MeshPoseModel(checkpoint=main_checkpoint)
        self.meshpose_model.main_model.to(device)
        self.branch_fusion = MeshPoseBranchFusion()
        self.upsampler = MeshUpsamplingDecoder(checkpoint=upsampler_checkpoint, device=device)

    def __call__(self, image, bbox):

        # Preprocess image
        input_data, trans, inv_trans = self.img_preprocessing(image, bbox)
        # Run model on image
        output = self.meshpose_model(input_data.unsqueeze(0).to(device))
        # Run the fusion decoder
        xyz_crop_lp, vertex_vis_lp = self.branch_fusion(output)
        vertex_vis_lp = vertex_vis_lp.squeeze(0).detach().cpu().numpy()
        # Run upsampling
        xyz_crop_hp = self.upsampler(xyz_crop_lp).squeeze(0).detach().cpu().numpy()
        # Run transform back to image
        xyz_im_hp = affine_tranform_3d(xyz_crop_hp, inv_trans)
        xyz_im_lp = affine_tranform_3d(xyz_crop_lp.squeeze(0).detach().cpu().numpy(), inv_trans)

        return {'vertex_vis_lp': vertex_vis_lp,
                'xyz_lp': xyz_im_lp,
                'xyz_hp': xyz_im_hp}
