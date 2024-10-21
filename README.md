# MeshPose: Unifying DensePose and 3D Body Mesh reconstruction

Inference and Evaluation code for the paper **MeshPose: Unifying DensePose and 3D Body Mesh reconstruction (CVPR 2024)**

[![report](https://img.shields.io/badge/Project-Page-blue)](https://meshpose.github.io/)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2406.10180)

![Example Image](assets/screenshots_from_video_demos.png)

## Installation

```
git clone https://github.com/Snapchat/MeshPose.git

conda create -n meshpose python=3.11
conda activate meshpose

pip install -r requirements.txt
```

Please download the model weights and place them in `./checkpoints` via the links in `./checkpoints/checkpoints.md`

The code has been tested on Ubuntu and Mac (on both GPU and CPU-only machines).

## Run Demo 
To run MeshPose on an image with a bounding box:
```
python3 inference.py
```
This will plot the front and side view of the predicted vertices on top of the original image.

## Run Video Demo
To run MeshPose on a video using a simple person detector/tracker:
```
python3 video_demo.py --input_video assets/4812014-hd_1920_1080_30fps.mp4 --do_rendering
```
This will render the meshes of all detected persons on top of the original video. 
It will also save the predicted vertices for every frame in a json-file.

To use the ``--do_rendering`` option, ``densepose_eval`` must be installed in the ``third_party`` directory (see below).

## Run Image Folder Demo
To run MeshPose on an image folder using a simple person detector/tracker:
```
python3 images_demo.py --input_folder assets/example_images --do_rendering
```
This will render the meshes of all detected persons on top of each original image. 
It will also save the predicted vertices for each image in a json-file.

To use the ``--do_rendering`` option, ``densepose_eval`` must be installed in the ``third_party`` directory (see below).

## MeshPose Evaluation on the DensePose Benchmark

### Data and Benchmark Preparation

Clone `densepose_eval` in `third_party`
```
cd third_party
git clone https://github.com/MeshPose/densepose_eval.git
```
and follow its [installation instructions](https://github.com/MeshPose/densepose_eval?tab=readme-ov-file#installation).

Download the `densepose minival` dataset and the UV data into `./DensePose_COCO` according to the instructions in [`./DensePose_COCO/densepose_dataset.md`](https://github.com/Snapchat/MeshPose/blob/main/DensePose_COCO/densepose_dataset.md).


### MeshPose Inference on Densepose Minival
The following command will run meshpose on each instance in the evaluation dataset and save the results in `output/model_predictions.json`
```
python3 inference_coco.py --output_model_predictions output/model_predictions.json
```

### Evaluation of MeshPose Mesh Alignment on the DensePose Benchmark
```
python3 evaluate_densepose.py --input_model_predictions output/model_predictions.json --output_densepose_score output/densepose_predictions.txt
```

## General Human Mesh Recovery Evaluation on the DensePose Benchmark
To evaluate another Human Mesh Recovery method on DensePose, create a json file (`my_mesh_predictions.json`) with the following structure:
```
[
            {'image_id': $image_id_0,  # int
             'id': $instance_id_0,  # int
             'smpl_z': $verts_z_0,  # (6980, )
             'smpl_xy_proj': $verts_xy_proj_0}  # (6980, 2),
             
            {'image_id': $image_id_1,  # int
             'id': $instance_id_1,  # int
             'smpl_z': $verts_z_1,  # (6980, )
             'smpl_xy_proj': $verts_xy_proj_1}  # (6980, 2),
        ...

]

```
This is a list of dictionaries, each dictionary corresponding to an instance in `DensePose_COCO/densepose_coco_2014_minival.json`.
```
image_id: the COCO image_id of the image containing the instance
id: the COCO id of the instance
smpl_z: a list containing the depth of each vertex (Normalized in (-1, 1))
smpl_xy_proj: a list of tuples (x,y) corresponding to the projection of the mesh on the original image
```

Please make sure that the `smpl_xy_proj` coordinates are aligned with the original image.

Note: To accelerate this, you can skip instances that don't have a `'dp_masks'` field, as they don't contain DensePose annotations and don't contribute to the metrics.

Once `my_mesh_predictions.json` is ready, the system can be evaluated via:

```
python3 evaluate_densepose.py --input_model_predictions my_mesh_predictions.json --output_densepose_score output/my_mesh_predictions.txt
```

Results are saved in `output/my_mesh_predictions.txt`. We report the `GPSM AR` and `GPSM AP` quantities.

## Citing
```
@InProceedings{Le_2024_CVPR,
    author    = {Le, Eric-Tuan and Kakolyris, Antonis and Koutras, Petros and Tam, Himmy and Skordos, Efstratios and Papandreou, George and G\"uler, Riza Alp and Kokkinos, Iasonas},
    title     = {MeshPose: Unifying DensePose and 3D Body Mesh Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {2405-2414}
}
```

## Contact

For questions about this work please contact [akakolyris@snap.com](akakolyris@snap.com) or [e.le@cs.ucl.ac.uk](e.le@cs.ucl.ac.uk)