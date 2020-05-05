## HueSaturationValue
```python
HueSaturationValue(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, hue_shift_limit:Union[int, Tuple[int, int]]=20, sat_shift_limit:Union[int, Tuple[int, int]]=30, val_shift_limit:Union[int, Tuple[int, int]]=20)
```
Randomly modify the hue, saturation, and value of an image.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **hue_shift_limit** :  Range for changing hue. If hue_shift_limit is a single int, the range will be            (-hue_shift_limit, hue_shift_limit).
* **sat_shift_limit** :  Range for changing saturation. If sat_shift_limit is a single int, the range will be            (-sat_shift_limit, sat_shift_limit).
* **val_shift_limit** :  range for changing value. If val_shift_limit is a single int, the range will be            (-val_shift_limit, val_shift_limit).
* **Image types** :         uint8, float32    