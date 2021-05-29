## NumpyOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/numpyop.py/#L29-L97>View source on Github</a>
```python
NumpyOp(
	inputs: Union[NoneType, str, Iterable[str]]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
An Operator class which takes and returns numpy data.

These Operators are used in fe.Pipeline to perform data pre-processing / augmentation. They may also be used in
fe.Network to perform postprocessing on data.


<h3>Args:</h3>


* **inputs**: Key(s) from which to retrieve data from the data dictionary.

* **outputs**: Key(s) under which to write the outputs of this Op back to the data dictionary.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

---

### forward<span class="tag">method of NumpyOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/numpyop.py/#L52-L66>View source on Github</a>
```python
forward(
	self,
	data: Union[numpy.ndarray, List[numpy.ndarray]],
	state: Dict[str, Any]
)
-> Union[numpy.ndarray, List[numpy.ndarray]]
```
A method which will be invoked in order to transform data.

This method will be invoked on individual elements of data before any batching / axis expansion is performed.


<h4>Args:</h4>


* **data**: The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.

* **state**: Information about the current execution context, for example {"mode": "train"}. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The <code>data</code> after applying whatever transform this Op is responsible for. It will be written into the data
    dictionary based on whatever keys this Op declares as its <code>outputs</code>.</li></ul>

---

### forward_batch<span class="tag">method of NumpyOp</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/numpyop.py/#L68-L97>View source on Github</a>
```python
forward_batch(
	self,
	data: Union[~Tensor, List[~Tensor]],
	state: Dict[str, Any]
)
-> Union[numpy.ndarray, List[numpy.ndarray]]
```
A method which will be invoked in order to transform a batch of data.

This method will be invoked on batches of data during network postprocessing. Note that the inputs may be numpy
arrays or TF/Torch tensors. Outputs are expected to be Numpy arrays, though this is not enforced. Developers
should probably not need to override this implementation unless they are building an op specifically intended
for postprocessing.


<h4>Args:</h4>


* **data**: The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.

* **state**: Information about the current execution context, for example {"mode": "train"}. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The <code>data</code> after applying whatever transform this Op is responsible for. It will be written into the data
    dictionary based on whatever keys this Op declares as its <code>outputs</code>.</li></ul>

