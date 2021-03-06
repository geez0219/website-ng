## Repeat<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/meta/repeat.py/#L25-L81>View source on Github</a>
```python
Repeat(
	op: fastestimator.op.numpyop.numpyop.NumpyOp,
	repeat: Union[int, Callable[..., bool]]=1,
	max_iter: Union[int, NoneType]=None
)
-> None
```
Repeat a NumpyOp several times in a row.


<h3>Args:</h3>


* **op**: A NumpyOp to be run one or more times in a row.

* **repeat**: How many times to repeat the `op`. This can also be a function return, in which case the function input names will be matched to keys in the data dictionary, and the `op` will be repeated until the function evaluates to False. The function evaluation will happen at the end of a forward call, so the `op` will always be evaluated at least once.

* **max_iter**: A limit to how many iterations will be run (or None for no limit). 

<h3>Raises:</h3>


* **ValueError**: If `repeat`, `op`, or max_iter are invalid.

