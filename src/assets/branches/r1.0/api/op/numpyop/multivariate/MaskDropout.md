## MaskDropout
```python
MaskDropout(max_objects:Union[int, Tuple[int, int]]=1, image_fill_value:Union[int, float, str]=0, mask_fill_value:Union[int, float]=0, mode:Union[str, NoneType]=None, image_in:Union[str, NoneType]=None, mask_in:Union[str, NoneType]=None, masks_in:Union[str, NoneType]=None, image_out:Union[str, NoneType]=None, mask_out:Union[str, NoneType]=None, masks_out:Union[str, NoneType]=None)
```
Zero out objects from an image + mask pair.    An image & mask augmentation that zero out mask and image regions corresponding to randomly chosen object instance    from mask. The mask must be single-channel image, with zero values treated as background. The image can be any    number of channels.

#### Args:

* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **image_in** :  The key of an image to be modified.
* **mask_in** :  The key of a mask to be modified (with the same random factors as the image).
* **masks_in** :  The key of masks to be modified (with the same random factors as the image).
* **image_out** :  The key to write the modified image (defaults to `image_in` if None).
* **mask_out** :  The key to write the modified mask (defaults to `mask_in` if None).
* **masks_out** :  The key to write the modified masks (defaults to `masks_in` if None).
* **max_objects** :  Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
* **image_fill_value** :  Fill value to use when filling image.            Can be 'inpaint' to apply in-painting (works only  for 3-channel images)
* **mask_fill_value** :  Fill value to use when filling mask.
* **Image types** :         uint8, float32    