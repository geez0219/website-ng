## OneOf
```python
OneOf(
	*tensor_ops: fastestimator.op.tensorop.tensorop.TensorOp
)
-> None
```
Perform one of several possible TensorOps.


#### Args:

* **tensor_ops** :  A list of ops to choose between with uniform probability.

### forward
```python
forward(
	self,
	data: Union[~Tensor, List[~Tensor]],
	state: Dict[str, Any]
)
-> Union[~Tensor, List[~Tensor]]
```
Execute a randomly selected op from the list of `numpy_ops`.


#### Args:

* **data** :  The information to be passed to one of the wrapped operators.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
    The `data` after application of one of the available numpyOps.