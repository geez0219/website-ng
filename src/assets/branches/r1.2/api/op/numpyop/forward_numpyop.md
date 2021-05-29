## forward_numpyop<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/numpyop.py/#L141-L167>View source on Github</a>
```python
forward_numpyop(
	ops: List[fastestimator.op.numpyop.numpyop.NumpyOp],
	data: MutableMapping[str, Any],
	state: Dict[str, Any],
	batched: bool=False
)
-> None
```
Call the forward function for list of NumpyOps, and modify the data dictionary in place.


<h3>Args:</h3>


* **ops**: A list of NumpyOps to execute.

* **data**: The data dictionary.

* **state**: Information about the current execution context, ex. {"mode": "train"}. Must contain at least the mode.

* **batched**: Whether the `data` is batched or not.

