## ISONoise
```python
ISONoise(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, color_shift:Tuple[float, float]=(0.01, 0.05), intensity:Tuple[float, float]=(0.1, 0.5))
```
Apply camera sensor noise.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **color_shift** :  Variance range for color hue change. Measured as a fraction of 360 degree Hue angle in the HLS            colorspace.
* **intensity** :  Multiplicative factor that controls the strength of color and luminace noise.
* **Image types** :         uint8    