## OneOf
```python
OneOf(*numpy_ops:fastestimator.op.numpyop.numpyop.NumpyOp) -> None
```
Perform one of several possible NumpyOps.

#### Args:

* **numpy_ops** :  A list of ops to choose between with uniform probability.    

### forward
```python
forward(self, data:Union[numpy.ndarray, List[numpy.ndarray]], state:Dict[str, Any]) -> Union[numpy.ndarray, List[numpy.ndarray]]
```
Execute a randomly selected op from the list of `numpy_ops`.

#### Args:

* **data** :  The information to be passed to one of the wrapped operators.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
            The `data` after application of one of the available numpyOps.        