## ElasticTransform
```python
ElasticTransform(alpha:float=34.0, sigma:float=4.0, alpha_affine:float=50.0, interpolation:int=1, border_mode:int=4, value:Union[NoneType, int, float, List[int], List[float]]=None, mask_value:Union[NoneType, int, float, List[int], List[float]]=None, approximate:bool=False, mode:Union[str, NoneType]=None, image_in:Union[str, NoneType]=None, mask_in:Union[str, NoneType]=None, masks_in:Union[str, NoneType]=None, image_out:Union[str, NoneType]=None, mask_out:Union[str, NoneType]=None, masks_out:Union[str, NoneType]=None)
```
Elastic deformation of images.

#### Args:

* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **image_in** :  The key of an image to be modified.
* **mask_in** :  The key of a mask to be modified (with the same random factors as the image).
* **masks_in** :  The key of masks to be modified (with the same random factors as the image).
* **image_out** :  The key to write the modified image (defaults to `image_in` if None).
* **mask_out** :  The key to write the modified mask (defaults to `mask_in` if None).
* **masks_out** :  The key to write the modified masks (defaults to `masks_in` if None).
* **alpha** :  Scaling factor during point translation.
* **sigma** :  Gaussian filter parameter. The effect (small to large) is random -> elastic -> affine -> translation.
* **alpha_affine** :  The range will be (-alpha_affine, alpha_affine).
* **interpolation** :  Flag that is used to specify the interpolation algorithm. Should be one of            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
* **border_mode** :  Flag that is used to specify the pixel extrapolation method. Should be one of            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
* **value** :  Padding value if border_mode is cv2.BORDER_CONSTANT.
* **mask_value** :  Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
* **approximate** :  Whether to smooth displacement map with fixed kernel size. Enabling this option gives ~2X            speedup on large (512x512) images.
* **Image types** :         uint8, float32    