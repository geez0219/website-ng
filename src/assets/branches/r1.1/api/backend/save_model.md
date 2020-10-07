## save_model<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/save_model.py/#L23-L76>View source on Github</a>
```python
save_model(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	save_dir: str,
	model_name: Union[str, NoneType]=None,
	save_optimizer: bool=False
)
```
Save `model` weights to a specific directory.

This method can be used with TensorFlow models:
```python
m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.h5' file inside /tmp directory
```

This method can be used with PyTorch models:
```python
m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
fe.backend.save_model(m, save_dir="/tmp", model_name="test")  # Generates 'test.pt' file inside /tmp directory
```


<h3>Args:</h3>

* **model** :  A neural network instance to save.
* **save_dir** :  Directory into which to write the `model` weights.
* **model_name** :  The name of the model (used for naming the weights file). If None, model.model_name will be used.
* **save_optimizer** :  Whether to save optimizer. If True, optimizer will be saved in a separate file at same folder.

<h3>Returns:</h3>
    The saved model path.

<h3>Raises:</h3>

* **ValueError** :  If `model` is an unacceptable data type.

