## GaussianNoise
```python
GaussianNoise(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, var_limit:Union[float, Tuple[float, float]]=(10.0, 50.0), mean:float=0.0)
```
Apply gaussian noise to the image

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **var_limit** :  Variance range for noise. If var_limit is a single float, the range will be (0, var_limit).
* **mean** :  Mean of the noise.
* **Image types** :         uint8, float32    