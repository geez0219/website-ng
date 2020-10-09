## ToFloat<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/to_float.py/#L22-L41>View source on Github</a>
```python
ToFloat(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	max_value: Union[float, NoneType]=None
)
```
Divides an input by max_value to give a float image in range [0,1].


<h3>Args:</h3>


* **inputs**: Key(s) of images to be converted to floating point representation.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **max_value**: The maximum value to serve as the divisor. If None it will be inferred by dtype. Image types: Any

