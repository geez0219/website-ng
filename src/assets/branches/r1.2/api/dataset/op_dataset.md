# tf.dataset.op_dataset<span class="tag">module</span>

---

## OpDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/op_dataset.py/#L88-L149>View source on Github</a>
```python
OpDataset(
	dataset: torch.utils.data.dataset.Dataset,
	ops: List[fastestimator.op.numpyop.numpyop.NumpyOp],
	mode: str,
	output_keys: Union[Set[str], NoneType]=None,
	deep_remainder: bool=True
)
-> None
```
A wrapper for datasets which allows operators to be applied to them in a pipeline.

This class should not be directly instantiated by the end user. The fe.Pipeline will automatically wrap datasets
within an Op dataset as needed.


<h3>Args:</h3>


* **dataset**: The base dataset to wrap.

* **ops**: A list of ops to be applied after the base `dataset` `__getitem__` is invoked.

* **mode**: What mode the system is currently running in ('train', 'eval', 'test', or 'infer').

* **output_keys**: What keys can be produced from pipeline. If None, all keys will be considered.

* **deep_remainder**: Whether data which is not modified by Ops should be deep copied or not. This argument is used to help with RAM management, but end users can almost certainly ignore it.

