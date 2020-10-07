## Sometimes<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/sometimes.py/#L29-L90>View source on Github</a>
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


<h3>Args:</h3>

* **tensor_op** :  The operator to be performed.
* **prob** :  The probability of execution, which should be in the range [0-1).

### forward<span class="tag">method of Sometimes</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/sometimes.py/#L71-L90>View source on Github</a>
```python
forward(
	self,
	data: List[~Tensor],
	state: Dict[str, Any]
)
-> List[~Tensor]
```
Execute the wrapped operator a certain fraction of the time.


<h4>Args:</h4>

* **data** :  The information to be passed to the wrapped operator.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

<h4>Returns:</h4>
    The original `data`, or the `data` after running it through the wrapped operator.



