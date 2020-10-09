## Reshape<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/tensorop/reshape.py/#L25-L52>View source on Github</a>
```python
Reshape(
	inputs: Union[str, List[str]],
	outputs: Union[str, List[str]],
	shape: Union[int, Tuple[int, ...]],
	mode: Union[NoneType, str, Iterable[str]]='!infer'
)
-> None
```
Reshape a input tensor to conform to a given shape.


<h3>Args:</h3>


* **inputs**: Key of the input tensor that is to be reshaped.

* **outputs**: Key of the output tensor that has been reshaped.

* **shape**: Target shape.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

