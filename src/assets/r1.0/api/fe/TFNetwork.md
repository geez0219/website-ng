## TFNetwork
```python
TFNetwork(ops:Iterable[Union[fastestimator.op.tensorop.tensorop.TensorOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.tensorop.tensorop.TensorOp]]]) -> None
```


### transform
```python
transform(self, data:Dict[str, Any], mode:str, epoch:int=1) -> Dict[str, Any]
```
apply all network operations on given data for certain mode and epoch.

#### Args:

* **data** :  Input data in dictionary format
* **mode** :  Current mode, can be "train", "eval", "test" or "infer"
* **epoch** :  Current epoch index. Defaults to 1.

#### Returns:
            transformed data        