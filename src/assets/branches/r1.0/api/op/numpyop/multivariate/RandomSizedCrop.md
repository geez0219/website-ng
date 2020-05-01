## RandomSizedCrop
```python
RandomSizedCrop(min_max_height:Tuple[int, int], height:int, width:int, w2h_ratio:float=1.0, interpolation:int=1, mode:Union[str, NoneType]=None, image_in:Union[str, NoneType]=None, mask_in:Union[str, NoneType]=None, masks_in:Union[str, NoneType]=None, bbox_in:Union[str, NoneType]=None, keypoints_in:Union[str, NoneType]=None, image_out:Union[str, NoneType]=None, mask_out:Union[str, NoneType]=None, masks_out:Union[str, NoneType]=None, bbox_out:Union[str, NoneType]=None, keypoints_out:Union[str, NoneType]=None, bbox_params:Union[albumentations.core.composition.BboxParams, str, NoneType]=None, keypoint_params:Union[albumentations.core.composition.KeypointParams, str, NoneType]=None)
```
Crop a random part of the input and rescale it to some size.

#### Args:

* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
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
* **min_max_height** :  Crop size limits.
* **height** :  Height after crop and resize.
* **width** :  Width after crop and resize.
* **w2h_ratio** :  Aspect ratio of crop.
* **interpolation** :  flag that is used to specify the interpolation algorithm. Should be one of            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
* **Image types** :         uint8, float32    