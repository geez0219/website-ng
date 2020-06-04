## TorchNetwork
```python
TorchNetwork(ops:Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]]) -> None
```
An extension of BaseNetwork for PyTorch models.


#### Args:

* **ops** :  The ops defining the execution graph for this Network.

### load_epoch
```python
load_epoch(self, mode:str, epoch:int, output_keys:Union[Set[str], NoneType]=None, warmup:bool=False) -> None
```
Prepare the network to run a given epoch and mode.

This method is necessary since schedulers and op mode restrictions may result in different computation graphs
every epoch. This also moves all of the necessary models from the CPU onto the GPU(s).


#### Args:

* **mode** :  The mode to prepare to execute. One of 'train', 'eval', 'test', or 'infer'.
* **epoch** :  The epoch to prepare to execute.
* **output_keys** :  What keys must be moved from the GPU back to the CPU after executing a step.
* **warmup** :  Whether to prepare to execute it warmup mode or not (end users can likely ignore this argument).

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

### unload_epoch
```python
unload_epoch(self) -> None
```
Clean up the network after running an epoch.

In this case we move all of the models from the GPU(s) back to the CPU.