## GradientOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/tensorop/gradient/gradient.py/#L27-L58>View source on Github</a>
```python
GradientOp(
	inputs: Union[str, List[str]],
	finals: Union[str, List[str]],
	outputs: Union[str, List[str]],
	mode: Union[NoneType, str, Iterable[str]]='eval'
)
```
Return the gradients of finals w.r.t. inputs.


<h3>Args:</h3>


* **inputs**: The tensor(s) to compute gradients with respect to.

* **finals**: The tensor(s) to compute gradients from.

* **outputs**: The key(s) under which to save the gradients.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

