## FromFloat<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/from_float.py/#L25-L49>View source on Github</a>
```python
FromFloat(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	max_value: Union[float, NoneType]=None,
	dtype: Union[str, numpy.dtype]='uint16'
)
```
Takes an input float image in range [0, 1.0] and then multiplies by `max_value` to get an int image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **max_value**: The maximum value to serve as the multiplier. If None it will be inferred by dtype.

* **dtype**: The data type to cast the output as. Image types: float32

