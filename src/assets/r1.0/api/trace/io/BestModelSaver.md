## BestModelSaver
```python
BestModelSaver(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], save_dir:str, metric:Union[str, NoneType]=None, save_best_mode:str='min') -> None
```
Save the weights of best model based on a given evaluation metric.

#### Args:

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the model.
* **metric** :  Eval metric name to monitor. If None, the model's loss will be used.
* **save_best_mode** :  Can be 'min' or 'max'.

#### Raises:

* **AssertionError** :  If a `metric` is not provided and it cannot be inferred from the `model`.
* **ValueError** :  If `save_best_mode` is an unacceptable string.    