## ToArray<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/to_array.py/#L24-L48>View source on Github</a>
```python
ToArray(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	dtype: Union[str, NoneType]=None
)
```
Convert data to a numpy array.


<h3>Args:</h3>


* **inputs**: Key(s) of the data to be converted.

* **outputs**: Key(s) into which to write the converted data.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **dtype**: The dtype to apply to the output array, or None to infer the type.

