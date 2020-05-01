

### load_model
```python
load_model(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], weights_path:str)
```
Load saved weights for a given model.
* **This method can be used with TensorFlow models** : ```pythonm = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")fe.backend.save_model(m, save_dir="tmp")fe.backend.load_model(m, weights_path="tmp/saved_model.h5")```
* **This method can be used with PyTorch models** : ```pythonm = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")fe.backend.save_model(m, save_dir="tmp")fe.backend.load_model(m, weights_path="tmp/saved_model.pt")```

#### Args:

* **model** :  A neural network instance to load.
* **weights_path** :  Path to the `model` weights.

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.