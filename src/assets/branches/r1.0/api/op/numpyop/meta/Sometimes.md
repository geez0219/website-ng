## Sometimes
```python
Sometimes(numpy_op:fastestimator.op.numpyop.numpyop.NumpyOp, prob:float=0.5) -> None
```
Perform a NumpyOp with a given probability.

#### Args:

* **numpy_op** :  The operator to be performed.
* **prob** :  The probability of execution, which should be in the range [0-1).    

### forward
```python
forward(self, data:Union[numpy.ndarray, List[numpy.ndarray]], state:Dict[str, Any]) -> Union[numpy.ndarray, List[numpy.ndarray]]
```
Execute the wrapped operator a certain fraction of the time.

#### Args:

* **data** :  The information to be passed to the wrapped operator.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
            The original `data`, or the `data` after running it through the wrapped operator.        