## Brightness<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/brightness.py/#L27-L61>View source on Github</a>
```python
Brightness(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	limit: float=0.54
)
```
Randomly change the brightness of an image.

This is a wrapper for functionality provided by the PIL library:
https://github.com/python-pillow/Pillow/tree/master/src/PIL.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **limit**: Factor range for changing brightness. If limit is a single float, the range will be (-limit, limit). A factor of 0.0 gives a black image and a factor of 1.0 gives the original image. Image types: uint8

