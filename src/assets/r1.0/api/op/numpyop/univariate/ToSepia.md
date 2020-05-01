## ToSepia
```python
ToSepia(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None)
```
Convert an RGB image to sepia.

#### Args:

* **inputs** :  Key(s) of images to be converted to sepia.
* **outputs** :  Key(s) into which to write the sepia images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **Image types** :         uint8, float32    