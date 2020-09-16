## Affine
```python
Affine(*args, **kwargs)
```
Perform affine transformations on an image.


#### Args:

* **rotate** :  How much to rotate an image (in degrees). If a single value is given then images will be rotated by            a value sampled from the range [-n, n]. If a tuple (a, b) is given then each image will be rotated            by a value sampled from the range [a, b].
* **scale** :  How much to scale an image (in percentage). If a single value is given then all images will be scaled            by a value drawn from the range [1.0, n]. If a tuple (a,b) is given then each image will be scaled            based on a value drawn from the range [a,b].
* **shear** :  How much to shear an image (in degrees). If a single value is given then all images will be sheared            on X and Y by two values sampled from the range [-n, n]. If a tuple (a, b) is given then images will            be sheared on X and Y by two values randomly sampled from the range [a, b].
* **translate** :  How much to translate an image. If a single value is given then the translation extent will be            sampled from the range [0,n]. If a tuple (a,b) is given then the extent will be sampled from            the range [a,b]. If integers are given then the translation will be in pixels. If a float then            it will be as a fraction of the image size.
* **border_handling** :  What to do in order to fill newly created pixels. Options are 'constant', 'edge',            'symmetric', 'reflect', and 'wrap'. If a list is given, then the method will be randomly            selected from the options in the list.
* **fill_value** :  What pixel value to insert when border_handling is 'constant'.
* **interpolation** :  What interpolation method to use. Options (from fast to slow) are 'nearest_neighbor',            'bilinear', 'bicubic', 'biquartic', and 'biquintic'.
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
* **Image types** :     uint8, float32