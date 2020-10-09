## LambdaOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/numpyop.py/#L63-L84>View source on Github</a>
```python
LambdaOp(
	fn: Callable,
	inputs: Union[NoneType, str, Iterable[str]]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
```
An Operator that performs any specified function as forward function.


<h3>Args:</h3>


* **fn**: The function to be executed.

* **inputs**: Key(s) from which to retrieve data from the data dictionary.

* **outputs**: Key(s) under which to write the outputs of this Op back to the data dictionary.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

