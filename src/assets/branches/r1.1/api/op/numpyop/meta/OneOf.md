## OneOf<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/meta/one_of.py/#L25-L57>View source on Github</a>
```python
OneOf(
	*numpy_ops: fastestimator.op.numpyop.numpyop.NumpyOp
)
-> None
```
Perform one of several possible NumpyOps.


<h3>Args:</h3>

* **numpy_ops** :  A list of ops to choose between with uniform probability.

### forward<span class="tag">method of OneOf</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/meta/one_of.py/#L46-L57>View source on Github</a>
```python
forward(
	self,
	data: Union[numpy.ndarray, List[numpy.ndarray]],
	state: Dict[str, Any]
)
-> Union[numpy.ndarray, List[numpy.ndarray]]
```
Execute a randomly selected op from the list of `numpy_ops`.


<h4>Args:</h4>

* **data** :  The information to be passed to one of the wrapped operators.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

<h4>Returns:</h4>
    The `data` after application of one of the available numpyOps.



