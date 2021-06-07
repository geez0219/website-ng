## Network<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/network.py/#L296-L341>View source on Github</a>
```python
Network(
	ops: Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]],
	pops: Union[NoneType, fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp], Iterable[Union[fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp]]]]=None
)
-> fastestimator.network.BaseNetwork
```
A function to automatically instantiate the correct Network derived class based on the given `ops`.


<h3>Args:</h3>


* **ops**: A collection of Ops defining the graph for this Network. It should contain at least one ModelOp, and all models should be either TensorFlow or Pytorch. We currently do not support mixing TensorFlow and Pytorch models within the same network.

* **pops**: Postprocessing Ops. A collection of NumpyOps to be run on the CPU after all of the normal `ops` have been executed. Unlike the NumpyOps found in the pipeline, these ops will run on batches of data rather than single points. 

<h3>Raises:</h3>


* **AssertionError**: If TensorFlow and PyTorch models are mixed, or if no models are provided.

* **ValueError**: If a model is provided whose type cannot be identified as either TensorFlow or PyTorch.

<h3>Returns:</h3>

<ul class="return-block"><li>    A network instance containing the given <code>ops</code>.

</li></ul>

