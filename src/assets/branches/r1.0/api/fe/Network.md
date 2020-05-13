

### Network
```python
Network(ops:Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]]) -> network.BaseNetwork
```
A function to automatically instantiate the correct Network derived class based on the given `ops`.


#### Args:

* **ops** :  A collection of Ops defining the graph for this Network. It should contain at least one ModelOp, and all        models should be either TensorFlow or Pytorch. We currently do not support mixing TensorFlow and Pytorch        models within the same network.

#### Returns:
    A network instance containing the given `ops`.

#### Raises:

* **AssertionError** :  If TensorFlow and PyTorch models are mixed, or if no models are provided.
* **ValueError** :  If a model is provided whose type cannot be identified as either TensorFlow or PyTorch.