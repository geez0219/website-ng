## Fuse
```python
Fuse(
	ops: Union[fastestimator.op.tensorop.tensorop.TensorOp, List[fastestimator.op.tensorop.tensorop.TensorOp]]
)
-> None
```
Run a sequence of TensorOps as a single Op.


#### Args:

* **ops** :  A sequence of TensorOps to run. They must all share the same mode. It also doesn't support scheduled ops at        the moment, though the subnet itself may be scheduled.

#### Raises:

* **ValueError** :  If `ops` are invalid.