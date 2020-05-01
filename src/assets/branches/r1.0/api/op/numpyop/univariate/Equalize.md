## Equalize
```python
Equalize(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, eq_mode:str='cv', by_channels:bool=True, mask:Union[NoneType, numpy.ndarray, Callable]=None, mask_params:List[str]=())
```
Equalize the image histogram.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **eq_mode** :  {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
* **by_channels** :  If True, use equalization by channels separately, else convert image to YCbCr representation and            use equalization by `Y` channel.
* **mask** :  If given, only the pixels selected by the mask are included in the analysis. May be 1 channel or 3 channel            array or callable. Function signature must include `image` argument.
* **mask_params** :  Params for mask function.
* **Image types** :         uint8    