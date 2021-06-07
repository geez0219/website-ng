## Repeat<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/repeat.py/#L30-L163>View source on Github</a>
```python
Repeat(
	op: fastestimator.op.tensorop.tensorop.TensorOp,
	repeat: Union[int, Callable[..., bool]]=1
)
-> None
```
Repeat a TensorOp several times in a row.


<h3>Args:</h3>


* **op**: A TensorOp to be run one or more times in a row.

* **repeat**: How many times to repeat the `op`. This can also be a function return, in which case the function input names will be matched to keys in the data dictionary, and the `op` will be repeated until the function evaluates to False. The function evaluation will happen at the end of a forward call, so the `op` will always be evaluated at least once. 

<h3>Raises:</h3>


* **ValueError**: If `repeat` or `op` are invalid.

