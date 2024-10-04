Download the DensePose dataset from http://densepose.org/#dataset and prepare into the following structure:
```
./DensePose_COCO/
  densepose_coco_2014_minival.json
  val2014/
    # image files that are mentioned in the minival json
```

You will also need the DensePose UV data, please follow the instructions from [here](https://github.com/facebookresearch/DensePose/blob/main/INSTALL.md#fetch-densepose-data).

```
./DensePose_COCO/
  densepose_coco_2014_minival.json
  val2014/
    # image files that are mentioned in the minival json
  densepose_uv_data/
    UV_Processed.mat
    ...
```