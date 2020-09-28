## Sometimes
```python
Sometimes(
	tensor_op: fastestimator.op.tensorop.tensorop.TensorOp,
	prob: float=0.5
)
-> None
```
Perform a NumpyOp with a given probability.

Note that Sometimes should not be used to wrap an op whose output key(s) do not already exist in the data
dictionary. This would result in a problem when future ops / traces attempt to reference the output key, but
Sometimes declined to generate it. If you want to create a default value for a new key, simply use a LambdaOp before
invoking the Sometimes.


#### Args:

* **tensor_op** :  The operator to be performed.
* **prob** :  The probability of execution, which should be in the range [0-1).

### forward
```python
forward(
	self,
	data: List[~Tensor],
	state: Dict[str, Any]
)
-> List[~Tensor]
```
Execute the wrapped operator a certain fraction of the time.


#### Args:

* **data** :  The information to be passed to the wrapped operator.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
    The original `data`, or the `data` after running it through the wrapped operator.