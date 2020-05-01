## Downscale
```python
Downscale(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, scale_min:float=0.25, scale_max:float=0.25, interpolation:int=0)
```
Decrease image quality by downscaling and then upscaling.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **scale_min** :  Lower bound on the image scale. Should be < 1.
* **scale_max** :   Upper bound on the image scale. Should be >= scale_min.
* **interpolation** :  cv2 interpolation method.
* **Image types** :         uint8, float32    