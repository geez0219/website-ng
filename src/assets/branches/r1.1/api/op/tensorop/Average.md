## Average<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/average.py/#L28-L47>View source on Github</a>
```python
Average(
	inputs: Union[str, Iterable[str]],
	outputs: str,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Compute the average across tensors.


<h3>Args:</h3>


* **inputs**: Keys of tensors to be averaged.

* **outputs**: The key under which to save the output.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

