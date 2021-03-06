## RandomCropNearBBox<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/multivariate/random_crop_near_bbox.py/#L25-L80>View source on Github</a>
```python
RandomCropNearBBox(
	cropping_bbox_in: str,
	max_part_shift: float=0.3,
	mode: Union[str, NoneType]=None,
	image_in: Union[str, NoneType]=None,
	mask_in: Union[str, NoneType]=None,
	masks_in: Union[str, NoneType]=None,
	bbox_in: Union[str, NoneType]=None,
	keypoints_in: Union[str, NoneType]=None,
	image_out: Union[str, NoneType]=None,
	mask_out: Union[str, NoneType]=None,
	masks_out: Union[str, NoneType]=None,
	bbox_out: Union[str, NoneType]=None,
	keypoints_out: Union[str, NoneType]=None,
	bbox_params: Union[albumentations.core.composition.BboxParams, str, NoneType]=None,
	keypoint_params: Union[albumentations.core.composition.KeypointParams, str, NoneType]=None
)
```
Crop bbox from an image with random shift by x,y coordinates.


<h3>Args:</h3>


* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **image_in**: The key of an image to be modified.

* **cropping_bbox_in**: The key of the cropping box, in [x1, y1, x2, y2] format.

* **mask_in**: The key of a mask to be modified (with the same random factors as the image).

* **masks_in**: The key of masks to be modified (with the same random factors as the image).

* **bbox_in**: The key of a bounding box(es) to be modified (with the same random factors as the image).

* **keypoints_in**: The key of keypoints to be modified (with the same random factors as the image).

* **image_out**: The key to write the modified image (defaults to `image_in` if None).

* **mask_out**: The key to write the modified mask (defaults to `mask_in` if None).

* **masks_out**: The key to write the modified masks (defaults to `masks_in` if None).

* **bbox_out**: The key to write the modified bounding box(es) (defaults to `bbox_in` if None).

* **keypoints_out**: The key to write the modified keypoints (defaults to `keypoints_in` if None).

* **bbox_params**: Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').

* **keypoint_params**: Parameters defining the type of keypoints ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa').

* **max_part_shift**: Float value in the range (0.0, 1.0). Image types: uint8, float32

