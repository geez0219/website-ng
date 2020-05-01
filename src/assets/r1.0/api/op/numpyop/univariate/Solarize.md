## Solarize
```python
Solarize(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, threshold:Union[int, Tuple[int, int], float, Tuple[float, float]]=128)
```
Invert all pixel values above a threshold.

#### Args:

* **inputs** :  Key(s) of images to be solarized.
* **outputs** :  Key(s) into which to write the solarized images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **threshold** :  Range for the solarizing threshold. If threshold is a single value 't', the range will be [t, t].
* **Image types** :         uint8, float32    