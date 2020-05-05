

### get_lr
```python
get_lr(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module]) -> float
```
Get the learning rate of a given model.
* **This method can be used with TensorFlow models** : ```pythonm = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")b = fe.backend.get_lr(model=m)  # 0.001```
* **This method can be used with PyTorch models** : ```pythonm = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")b = fe.backend.get_lr(model=m)  # 0.001```

#### Args:

* **model** :  A neural network instance to inspect.

#### Returns:
    The learning rate of `model`.

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.