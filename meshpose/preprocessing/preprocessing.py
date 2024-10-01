import cv2
import torchvision.transforms as transforms
from meshpose.utils.utils import gen_trans_from_patch_cv, framebbox


class ImagePreprocessing:
    def __init__(self, crop_size=(288, 384), scale_bbox=1.1):
        self.crop_width, self.crop_height = crop_size
        self.scale_bbox = scale_bbox
        self.target_dsize = (self.crop_width, self.crop_height)
        self.aspect_ratio = self.crop_width * 1.0 / self.crop_height
        self.pad_value = [127.5, 0, 0]
        self.rgb_mean = [127.5 / 255.0] * 3
        self.rgb_std = [127.5 / 255.0] * 3
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )
        self.to_tensor = transform

    def __call__(self, image, bbox):
        bbox_center_x, bbox_center_y, bbox_w, bbox_h = framebbox(bbox, self.aspect_ratio, scale_bbox=self.scale_bbox)

        trans, inv_trans = gen_trans_from_patch_cv(c_x=bbox_center_x, c_y=bbox_center_y,
                                                   src_w=bbox_w, src_h=bbox_h,
                                                   dst_w=self.crop_width, dst_h=self.crop_height)

        cropped_image = cv2.warpAffine(src=image, M=trans, dsize=self.target_dsize, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=self.pad_value)

        return self.to_tensor(cropped_image), trans, inv_trans
