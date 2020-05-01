

### save_model
```python
save_model(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], save_dir:str, model_name:str='saved_model')
```
Save `model` weights to a specific directory.
* **This method can be used with TensorFlow models** : ```pythonm = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.h5' file inside /tmp directory```
* **This method can be used with PyTorch models** : ```pythonm = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.pt' file inside /tmp directory```

#### Args:

* **model** :  A neural network instance to save.
* **save_dir** :  Directory into which to write the `model` weights.
* **model_name** :  The name of the model (used for naming the weights file).

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.