import os
import json
import argparse
from tqdm import tqdm
import cv2

from meshpose.utils.detector_inference import PersonDetector
from meshpose.utils.meshpose_inference import MeshPoseInference
from meshpose.utils import round_np, visualize_vertices
from meshpose.postprocessing.mesh_renderer import MeshRenderer


def process_video(input_video, output_video, do_rendering=True):
    # Open input video.
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Open output video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Create detector-tracking model.
    detector = PersonDetector(momentum=0.6)
    # Create MeshPose model.
    meshpose = MeshPoseInference()
    # Create mesh renderer.
    renderer = MeshRenderer((width, height)) if do_rendering else None

    model_predictions = list()
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[..., :3]
            bboxes = detector(frame)

            outputs = list()
            vertices = list()
            for bbox_ in bboxes:
                x1, y1, x2, y2 = bbox_
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                outputs_ = meshpose(frame, bbox_xywh)
                outputs_list_ = {key: round_np(item).tolist() for key, item in outputs_.items()}
                outputs.append(outputs_list_)
                vertices.append(outputs_['xyz_hp'])

            for bbox_ in bboxes:
                x1, y1, x2, y2 = bbox_.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if renderer is not None:
                frame = renderer(frame, vertices)
            else:
                frame = visualize_vertices(frame, outputs, vertices_type='xyz_lp')

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            model_predictions.append(outputs)
            pbar.update(1)

    cap.release()
    out.release()

    return model_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_videos')
    parser.add_argument('--do_rendering', action='store_true')
    args = parser.parse_args()

    input_video = args.input_video
    video_name = os.path.basename(input_video).split('.')[0]
    output_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, f'{video_name}.mp4')
    output_model_predictions = os.path.join(output_dir, f'{video_name}.json')

    model_predictions = process_video(input_video, output_video, args.do_rendering)
    with open(output_model_predictions, 'w') as f:
        json.dump(model_predictions, f)
