## Normalize
```python
Normalize(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, mean:Union[float, Tuple[float, ...]]=(0.485, 0.456, 0.406), std:Union[float, Tuple[float, ...]]=(0.229, 0.224, 0.225), max_pixel_value:float=255.0)
```
Divide pixel values by a maximum value, subtract mean per channel and divide by std per channel.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **mean** :  Mean values to subtract.
* **std** :  The divisor.
* **max_pixel_value** :  Maximum possible pixel value.
* **Image types** :         uint8, float32    