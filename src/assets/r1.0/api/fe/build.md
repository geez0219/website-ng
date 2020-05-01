

### build
```python
build(model_fn:Callable, optimizer_fn:Union[str, fastestimator.schedule.schedule.Scheduler, Callable, List[str], List[Callable], List[fastestimator.schedule.schedule.Scheduler], NoneType], weights_path:Union[str, NoneType, List[Union[str, NoneType]]]=None, model_names:Union[str, List[str], NoneType]=None) -> Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module, List[tensorflow.python.keras.engine.training.Model], List[torch.nn.modules.module.Module]]
```
Build model instances and associate them with optimizers

#### Args:

* **model_fn** :  function that define model(s)
* **optimizer_fn** :  optimizer string/definition or list of optimizer instances/strings. For example
* **tensorflow user can do optimizer_fn = lambda** :  tf.optimizers.Adam(lr=0.1),
* **pytorch user can do  optimizer_fn = lambda x** :  torch.optim.Adam(params=x, lr=0.1)
* **model_names** :  names of the model that will be used for logging purpose. If None, name will be assigned.
* **weights_path** :  weights path to load from. Defaults None.

#### Returns:

* **models** :  model(s) built by FastEstimator