import os
import cv2
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from meshpose.preprocessing import ImagePreprocessing
from meshpose.utils import affine_tranform_3d, read_dp_mask
from meshpose.postprocessing.uv_renderer import IUVRenderer


try:
    sys.path.insert(0, 'third_party/densepose_eval')
    from evaluator import _evaluate_predictions_on_coco, DensePoseChartResultQuantized
except ImportError as e:
    raise ImportError('Please download and install densepose_eval in ./third_party')


RENDER_RESOLUTION = (1024, 1024)


def evaluate_densepose(model_predictions, densepose_coco_minival, output_densepose_score):
    renderer = IUVRenderer(resolution=RENDER_RESOLUTION)
    img_preprocessing = ImagePreprocessing(crop_size=RENDER_RESOLUTION, scale_bbox=1.0)

    img_ids = list()
    densepose_predictions = list()

    coco_api = COCO(densepose_coco_minival)

    for prediction in tqdm(model_predictions):

        # Load coco data
        height = coco_api.imgs[prediction['image_id']]['height']
        width = coco_api.imgs[prediction['image_id']]['width']
        bbox = coco_api.anns[prediction['id']]['bbox']
        mask_enc = coco_api.anns[prediction['id']]['dp_masks']

        # Helpers for rendering
        _, trans, inv_trans = img_preprocessing(np.zeros((height, width, 3)), bbox)

        def crop_to_image(crop):
            return cv2.warpAffine(crop, inv_trans, dsize=(width, height),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # Load and predicted mesh
        pred_mesh_z = np.array(prediction['smpl_z'])
        pred_mesh_proj = np.array(prediction['smpl_xy_proj'])
        pred_mesh = np.concatenate((pred_mesh_proj, pred_mesh_z.reshape(-1, 1)), 1)
        xyz_im_hp = affine_tranform_3d(pred_mesh, trans)

        # Render IUV and segmentation map for mesh and transform to image space
        rgb_i, rgb_uv, segm = renderer(xyz_im_hp)

        img_i = crop_to_image(rgb_i)
        img_uv = crop_to_image(rgb_uv)
        segmentation = crop_to_image(segm)

        dense_uv = np.concatenate((img_i[:, :, :1], img_uv[:, :, :2]), axis=2)

        # Read annotated segmentation map
        x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
        x2, y2 = min(int(bbox[0]) + int(bbox[2]), width), min(int(bbox[1]) + int(bbox[3]), height)

        mask = read_dp_mask(mask_enc)
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        segmentation_gt = np.zeros([height, width])
        segmentation_gt[y1:y2, x1:x2] = mask > 0

        # Mask with the ground truth segmentation, crop and prepare to densepose benchmark form
        segmentation_masked = segmentation_gt * segmentation
        segmentation_enc = mask_util.encode(np.asfortranarray(np.uint8(segmentation_masked)))

        dense_uv = dense_uv[y1:y2, x1:x2]
        dense_uv_masked = np.uint8(dense_uv * np.expand_dims(segmentation_masked[y1:y2, x1:x2], axis=2))
        densepose_enc = DensePoseChartResultQuantized(torch.ByteTensor(dense_uv_masked).permute(2, 0, 1))

        prediction4densepose = {'image_id': prediction['image_id'],
                                'id': prediction['id'],
                                'category_id': 1,
                                'score': 1.0,
                                'bbox': bbox,
                                'dense_uv': dense_uv_masked,
                                'segmentation': segmentation_enc,
                                'densepose': densepose_enc}

        img_ids.append(prediction['image_id'])
        densepose_predictions.append(prediction4densepose)

    # Quantitative DensePose Evaluation
    results_densepose = _evaluate_predictions_on_coco(
        coco_api,
        densepose_predictions,
        None,
        None,
        class_names=['person'],
        min_threshold=0.5,
        img_ids=img_ids,
    )
    with open(output_densepose_score, 'w') as f:
        for name_metric, result_metric in zip(['GPS', 'GPSM', 'Segmentation'], results_densepose):
            f.write(name_metric)
            f.write('\n')
            f.write(str(result_metric))
            f.write('\n')
            print(name_metric)
            print(result_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_predictions', type=str, default='output/model_predictions.json')
    parser.add_argument('--output_densepose_score', type=str, default='output/densepose_predictions.txt')
    parser.add_argument('--densepose_coco_minival', type=str, default='DensePose_COCO/densepose_coco_2014_minival.json')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    print('Loading predictions...')
    with open(args.input_model_predictions, 'r') as f:
        predictions = json.load(f)
    print('Finished loading!')

    evaluate_densepose(predictions, args.densepose_coco_minival, args.output_densepose_score)
