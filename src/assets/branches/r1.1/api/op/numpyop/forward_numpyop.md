## forward_numpyop<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/numpyop.py/#L87-L102>View source on Github</a>
```python
forward_numpyop(
	ops: List[fastestimator.op.numpyop.numpyop.NumpyOp],
	data: MutableMapping[str, Any],
	mode: str
)
-> None
```
Call the forward function for list of NumpyOps, and modify the data dictionary in place.


<h3>Args:</h3>


* **ops**: A list of NumpyOps to execute.

* **data**: The data dictionary.

* **mode**: The current execution mode ("train", "eval", "test", or "infer").

