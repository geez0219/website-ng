## CoarseDropout
```python
CoarseDropout(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, max_holes:int=8, max_height:int=8, max_width:int=8, min_holes:Union[int, NoneType]=None, min_height:Union[int, NoneType]=None, min_width:Union[int, NoneType]=None, fill_value:Union[int, float, List[int], List[float]]=0)
```
Drop rectangular regions from an image.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **max_holes** :  Maximum number of regions to zero out.
* **max_height** :  Maximum height of the hole.
* **max_width** :  Maximum width of the hole.
* **min_holes** :  Minimum number of regions to zero out. If `None`, `min_holes` is set to `max_holes`.
* **min_height** :  Minimum height of the hole. If `None`, `min_height` is set to `max_height`.
* **min_width** :  Minimum width of the hole. If `None`, `min_height` is set to `max_width`.
* **fill_value** :  value for dropped pixels.
* **Image types** :         uint8, float32    