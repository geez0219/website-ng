## TFNetwork
```python
TFNetwork(ops:Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]]) -> None
```
An extension of BaseNetwork for TensorFlow models.
    

### run_step
```python
run_step(self, batch:Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]
```
Run a forward step through the Network on a batch of data.

Implementations of this method within derived classes should handle bringing the prediction data back from the
(multi-)GPU environment to the CPU. This method expects that Network.load_epoch() has already been invoked.


#### Args:

* **batch** :  The batch of data serving as input to the Network.

#### Returns:
    (batch_data, prediction_data)

### transform
```python
transform(self, data:Dict[str, Any], mode:str, epoch:int=1) -> Dict[str, Any]
```
Run a forward step through the Network on an element of data.


#### Args:

* **data** :  The element to data to use as input.
* **mode** :  The mode in which to run the transform. One of 'train', 'eval', 'test', or 'infer'.
* **epoch** :  The epoch in which to run the transform.

#### Returns:
    (batch_data, prediction_data)