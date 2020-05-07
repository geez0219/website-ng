

### load_model
```python
load_model(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], weights_path:str, load_optimizer:bool=False)
```
Load saved weights for a given model.

This method can be used with TensorFlow models:
```python
m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
fe.backend.save_model(m, save_dir="tmp", model_name="test")
fe.backend.load_model(m, weights_path="tmp/test.h5")
```

This method can be used with PyTorch models:
```python
m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
fe.backend.save_model(m, save_dir="tmp", model_name="test")
fe.backend.load_model(m, weights_path="tmp/test.pt")
```



#### Args:

* **model** :  A neural network instance to load.
* **weights_path** :  Path to the `model` weights.
* **load_optimizer** :  Whether to load optimizer. If True, then it will load &lt;weights_opt&gt; file in the path.

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.