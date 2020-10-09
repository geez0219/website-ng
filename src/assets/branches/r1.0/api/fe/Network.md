## Network<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/network.py/#L217-L251>View source on Github</a>
```python
Network(
	ops: Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]]
)
-> fastestimator.network.BaseNetwork
```
A function to automatically instantiate the correct Network derived class based on the given `ops`.


<h3>Args:</h3>


* **ops**: A collection of Ops defining the graph for this Network. It should contain at least one ModelOp, and all models should be either TensorFlow or Pytorch. We currently do not support mixing TensorFlow and Pytorch models within the same network. 

<h3>Raises:</h3>


* **AssertionError**: If TensorFlow and PyTorch models are mixed, or if no models are provided.

* **ValueError**: If a model is provided whose type cannot be identified as either TensorFlow or PyTorch.

<h3>Returns:</h3>

<ul class="return-block"><li>    A network instance containing the given <code>ops</code>.

</li></ul>

