## Fuse<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/fuse.py/#L30-L85>View source on Github</a>
```python
Fuse(
	ops: Union[fastestimator.op.tensorop.tensorop.TensorOp, List[fastestimator.op.tensorop.tensorop.TensorOp]]
)
-> None
```
Run a sequence of TensorOps as a single Op.


<h3>Args:</h3>


* **ops**: A sequence of TensorOps to run. They must all share the same mode. It also doesn't support scheduled ops at the moment, though the subnet itself may be scheduled. 

<h3>Raises:</h3>


* **ValueError**: If `ops` are invalid.

