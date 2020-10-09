## OneOf<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/one_of.py/#L30-L87>View source on Github</a>
```python
OneOf(
	*tensor_ops: fastestimator.op.tensorop.tensorop.TensorOp
)
-> None
```
Perform one of several possible TensorOps.


<h3>Args:</h3>


* **tensor_ops**: A list of ops to choose between with uniform probability.

---

### forward<span class="tag">method of OneOf</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/meta/one_of.py/#L76-L87>View source on Github</a>
```python
forward(
	self,
	data: Union[~Tensor, List[~Tensor]],
	state: Dict[str, Any]
)
-> Union[~Tensor, List[~Tensor]]
```
Execute a randomly selected op from the list of `numpy_ops`.


<h4>Args:</h4>


* **data**: The information to be passed to one of the wrapped operators.

* **state**: Information about the current execution context, for example {"mode": "train"}. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The <code>data</code> after application of one of the available numpyOps.</li></ul>

