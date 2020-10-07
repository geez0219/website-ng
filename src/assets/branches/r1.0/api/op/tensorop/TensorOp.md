## TensorOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/tensorop/tensorop.py/#L25-L43>View source on Github</a>
```python
TensorOp(
	inputs: Union[NoneType, str, Iterable[str], Callable]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
An Operator class which takes and returns tensor data.

These Operators are used in fe.Network to perform graph-based operations like neural network training.

### forward<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/tensorop/tensorop.py/#L30-L43>View source on Github</a>
```python
forward(
	self,
	data: Union[~Tensor, List[~Tensor]],
	state: Dict[str, Any]
)
-> Union[~Tensor, List[~Tensor]]
```
A method which will be invoked in order to transform data.

This method will be invoked on batches of data.


<h4>Args:</h4>

* **data** :  The batch from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

<h4>Returns:</h4>
    The `data` after applying whatever transform this Op is responsible for. It will be written into the data    dictionary based on whatever keys this Op declares as its `outputs`.



