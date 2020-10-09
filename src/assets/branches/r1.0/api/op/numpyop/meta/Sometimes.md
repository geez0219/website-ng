## Sometimes<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/meta/sometimes.py/#L22-L47>View source on Github</a>
```python
Sometimes(
	numpy_op: fastestimator.op.numpyop.numpyop.NumpyOp,
	prob: float=0.5
)
-> None
```
Perform a NumpyOp with a given probability.


<h3>Args:</h3>


* **numpy_op**: The operator to be performed.

* **prob**: The probability of execution, which should be in the range: [0-1).

---

### forward<span class="tag">method of Sometimes</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/meta/sometimes.py/#L34-L47>View source on Github</a>
```python
forward(
	self,
	data: Union[numpy.ndarray, List[numpy.ndarray]],
	state: Dict[str, Any]
)
-> Union[numpy.ndarray, List[numpy.ndarray]]
```
Execute the wrapped operator a certain fraction of the time.


<h4>Args:</h4>


* **data**: The information to be passed to the wrapped operator.

* **state**: Information about the current execution context, for example {"mode": "train"}. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The original <code>data</code>, or the <code>data</code> after running it through the wrapped operator.</li></ul>

