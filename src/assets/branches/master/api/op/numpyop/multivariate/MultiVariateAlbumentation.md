## MultiVariateAlbumentation
```python
MultiVariateAlbumentation(*args, **kwargs)
```
A base class for the DualTransform albumentation functions.

 DualTransforms are functions which apply simultaneously to images and corresponding information such as masks
 and/or bounding boxes.

This is a wrapper for functionality provided by the Albumentations library:
https://github.com/albumentations-team/albumentations. A useful visualization tool for many of the possible effects
it provides is available at https://albumentations-demo.herokuapp.com.


#### Args:

* **func** :  An Albumentation function to be invoked.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **image_in** :  The key of an image to be modified.
* **mask_in** :  The key of a mask to be modified (with the same random factors as the image).
* **masks_in** :  The key of masks to be modified (with the same random factors as the image).
* **bbox_in** :  The key of a bounding box(es) to be modified (with the same random factors as the image).
* **keypoints_in** :  The key of keypoints to be modified (with the same random factors as the image).
* **image_out** :  The key to write the modified image (defaults to `image_in` if None).
* **mask_out** :  The key to write the modified mask (defaults to `mask_in` if None).
* **masks_out** :  The key to write the modified masks (defaults to `masks_in` if None).
* **bbox_out** :  The key to write the modified bounding box(es) (defaults to `bbox_in` if None).
* **keypoints_out** :  The key to write the modified keypoints (defaults to `keypoints_in` if None).
* **bbox_params** :  Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').
* **keypoint_params** :  Parameters defining the type of keypoints ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa').

#### Raises:

* **AssertionError** :  If none of the various inputs such as `image_in` or `mask_in` are provided.