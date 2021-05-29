## TensorOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L28-L105>View source on Github</a>
```python
TensorOp(
	inputs: Union[NoneType, str, Iterable[str]]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
An Operator class which takes and returns tensor data.

These Operators are used in fe.Network to perform graph-based operations like neural network training.

---

### build<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L48-L59>View source on Github</a>
```python
build(
	self,
	framework: str,
	device: Union[torch.device, NoneType]=None
)
-> None
```
A method which will be invoked during Network instantiation.

This method can be used to augment the natural __init__ method of the TensorOp once the desired backend
framework is known.


<h4>Args:</h4>


* **framework**: Which framework this Op will be executing in. One of 'tf' or 'torch'.

* **device**: Which device this Op will execute on. Usually 'cuda:0' or 'cpu'. Only populated when the `framework` is 'torch'.

---

### fe_retain_graph<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L90-L105>View source on Github</a>
```python
fe_retain_graph(
	self,
	retain: Union[bool, NoneType]=None
)
-> Union[bool, NoneType]
```
A method to get / set whether this Op should retain network gradients after computing them.

All users and most developers can safely ignore this method. Ops which do not compute gradients should leave
this method alone. If this method is invoked with `retain` as True or False, then the gradient computations
performed by this Op should retain or discard the graph respectively afterwards.


<h4>Args:</h4>


* **retain**: If None, then return the current retain_graph status of the Op. If True or False, then set the retain_graph status of the op to the new status and return the new status. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Whether this Op will retain the backward gradient graph after it's forward pass, or None if this Op does not
    compute backward gradients.</li></ul>

---

### forward<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L33-L46>View source on Github</a>
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


* **data**: The batch from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.

* **state**: Information about the current execution context, for example {"mode": "train"}. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The <code>data</code> after applying whatever transform this Op is responsible for. It will be written into the data
    dictionary based on whatever keys this Op declares as its <code>outputs</code>.</li></ul>

---

### get_fe_loss_keys<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L78-L87>View source on Github</a>
```python
get_fe_loss_keys(
	self
)
-> Set[str]
```
A method to get any loss keys held by this Op.

All users and most developers can safely ignore this method. This method may be invoked to gather information
about losses, for example by the Network in get_loss_keys().


<h4>Returns:</h4>

<ul class="return-block"><li>    Any loss keys held by this Op.</li></ul>

---

### get_fe_models<span class="tag">method of TensorOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/tensorop.py/#L66-L75>View source on Github</a>
```python
get_fe_models(
	self
)
-> Set[~Model]
```
A method to get any models held by this Op.

All users and most developers can safely ignore this method. This method may be invoked to gather and manipulate
models, for example by the Network during load_epoch().


<h4>Returns:</h4>

<ul class="return-block"><li>    Any models held by this Op.</li></ul>

