

### set_lr
```python
set_lr(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], lr:float)
```
Set the learning rate of a given `model`.
* **This method can be used with TensorFlow models** : ```pythonm = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")  # m.optimizer.lr == 0.001fe.backend.set_lr(m, lr=0.8)  # m.optimizer.lr == 0.8```
* **This method can be used with PyTorch models** : ```pythonm = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")  # m.optimizer.param_groups[-1]['lr'] == 0.001fe.backend.set_lr(m, lr=0.8)  # m.optimizer.param_groups[-1]['lr'] == 0.8```

#### Args:

* **model** :  A neural network instance to modify.
* **lr** :  The learning rate to assign to the `model`.

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.