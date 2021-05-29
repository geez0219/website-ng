## TranslateY<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/translate_y.py/#L27-L64>View source on Github</a>
```python
TranslateY(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	shift_limit: float=0.2
)
```
Randomly shift the image along the Y axis.

This is a wrapper for functionality provided by the PIL library:
https://github.com/python-pillow/Pillow/tree/master/src/PIL.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **shift_limit**: Shift factor range as a fraction of image height. If shift_limit is a single float, the range will be (-shift_limit, shift_limit). Image types: uint8

