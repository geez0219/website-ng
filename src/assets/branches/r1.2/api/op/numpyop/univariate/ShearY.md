## ShearY<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/shear_y.py/#L27-L67>View source on Github</a>
```python
ShearY(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	shear_coef: float=0.3
)
```
Randomly shear the image along the Y axis.

This is a wrapper for functionality provided by the PIL library:
https://github.com/python-pillow/Pillow/tree/master/src/PIL.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **shear_coef**: Factor range for shear. If shear_coef is a single float, the range will be (-shear_coef, shear_coef) Image types: uint8

