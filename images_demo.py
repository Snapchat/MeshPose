import os
import json
import argparse
from tqdm import tqdm
import cv2

from meshpose.utils.detector_inference import PersonDetector
from meshpose.utils.meshpose_inference import MeshPoseInference
from meshpose.utils import imread, imwrite, round_np, visualize_vertices
from meshpose.postprocessing.mesh_renderer import MeshRenderer


def process_image_folder(input_folder, output_folder, do_rendering=True):
    # Create detector-tracking model.
    detector = PersonDetector(momentum=0.0)
    # Create MeshPose model.
    meshpose = MeshPoseInference()

    for image_file in tqdm(sorted(os.listdir(input_folder)), desc="Processing Image", unit="image"):
        image_path = os.path.join(input_folder, image_file)
        if not os.path.isfile(image_path):
            continue

        image = imread(image_path)
        image_width, image_height = image.shape[1], image.shape[0]

        bboxes = detector(image)

        outputs = list()
        vertices = list()
        for bbox_ in bboxes:
            x1, y1, x2, y2 = bbox_
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            outputs_ = meshpose(image, bbox_xywh)
            outputs_list_ = {key: round_np(item).tolist() for key, item in outputs_.items()}
            outputs.append(outputs_list_)
            vertices.append(outputs_['xyz_hp'])

        for bbox_ in bboxes:
            x1, y1, x2, y2 = bbox_.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Create mesh renderer.
        if do_rendering:
            renderer = MeshRenderer((image_width, image_height))
            image = renderer(image, vertices)
        else:
            image = visualize_vertices(image, outputs, vertices_type='xyz_lp')

        output_path = os.path.join(output_folder, image_file)
        imwrite(output_path, image)
        output_path_json = output_path.split(".")[0] + ".json"
        with open(output_path_json, 'w') as f:
            json.dump(outputs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_images')
    parser.add_argument('--do_rendering', action='store_true')
    args = parser.parse_args()

    input_folder = args.input_folder
    folder_name = os.path.basename(input_folder).split('.')[0]
    output_folder = os.path.join(args.output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    process_image_folder(input_folder, output_folder, args.do_rendering)
