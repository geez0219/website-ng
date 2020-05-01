## MedianBlur
```python
MedianBlur(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, blur_limit:Union[int, Tuple[int, int]]=5)
```
Blur the image with median filter of random aperture size.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **blur_limit** :  maximum aperture linear size for blurring the input image. Should be odd and in range [3, inf).            If image is a float type then only 3 and 5 are valid sizes.
* **Image types** :         uint8, float32    