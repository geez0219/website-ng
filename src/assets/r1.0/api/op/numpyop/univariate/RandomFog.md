## RandomFog
```python
RandomFog(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, fog_coef_lower:float=0.3, fog_coef_upper:float=1.0, alpha_coef:float=0.08)
```
Add fog to an image.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **fog_coef_lower** :  Lower limit for fog intensity coefficient. Should be in the range [0, 1].
* **fog_coef_upper** :  Upper limit for fog intensity coefficient. Should be in the range [0, 1].
* **alpha_coef** :  Transparency of the fog circles. Should be in the range [0, 1].
* **Image types** :         uint8, float32    