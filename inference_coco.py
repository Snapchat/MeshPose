import os
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO

from inference import MeshPoseInference
from meshpose.utils import imread, round_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--densepose_coco_minival', type=str, default='DensePose_COCO/densepose_coco_2014_minival.json')
    parser.add_argument('--root_img_dir', type=str, default='DensePose_COCO/val2014')
    parser.add_argument('--output_model_predictions', type=str, default='output/model_predictions.json')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    coco_api = COCO(args.densepose_coco_minival)

    meshpose = MeshPoseInference(scale_bbox=1.25)
    model_predictions = []

    for image_id, img in tqdm(coco_api.imgs.items()):
        img_filename = os.path.join(args.root_img_dir, img['file_name'])
        image = imread(img_filename)

        instances = coco_api.imgToAnns[image_id]

        for instance in instances:
            if 'dp_masks' not in instance:
                continue

            # image + bbox to image aligned mesh vertices
            outputs = meshpose(image, instance['bbox'])

            verts_z = round_np(outputs['xyz_hp'][:, 2]).tolist()  # depth of 3D vertices (in meters, pixels)
            verts_xy_proj = round_np(outputs['xyz_hp'][:, :2]).tolist()  # projected xyz (image space)

            model_predictions.append({'image_id': image_id,  # int
                                      'id': instance['id'],  # int
                                      'smpl_z': verts_z,  # (6980, )
                                      'smpl_xy_proj': verts_xy_proj}  # (6980, 2)
                                     )

    with open(args.output_model_predictions, 'w') as f:
        json.dump(model_predictions, f)
