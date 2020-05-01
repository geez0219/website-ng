## RandomSnow
```python
RandomSnow(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, snow_point_lower:float=0.1, snow_point_upper:float=0.3, brightness_coeff:float=2.5)
```
Bleach out some pixels to simulate snow.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **snow_point_lower** :  Lower bound of the amount of snow. Should be in the range [0, 1].
* **snow_point_upper** :  Upper bound of the amount of snow. Should be in the range [0, 1].
* **brightness_coeff** :  A larger number will lead to a more snow on the image. Should be >= 0.
* **Image types** :         uint8, float32    