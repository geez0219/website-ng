## OpDataset
```python
OpDataset(dataset:torch.utils.data.dataset.Dataset, ops:List[fastestimator.op.numpyop.numpyop.NumpyOp], mode:str) -> None
```
A wrapper for datasets which allows operators to be applied to them in a pipeline.    This class should not be directly instantiated by the end user. The fe.Pipeline will automatically wrap datasets    within an Op dataset as needed.

#### Args:

* **dataset** :  The base dataset to wrap.
* **ops** :  A list of ops to be applied after the base `dataset` __getitem__ is invoked.
* **mode** :  What mode the system is currently running in ('train', 'eval', 'test', or 'infer').    