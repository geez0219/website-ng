## get_lr<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/backend/get_lr.py/#L21-L51>View source on Github</a>
```python
get_lr(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module]
)
-> float
```
Get the learning rate of a given model.

This method can be used with TensorFlow models:
```python
m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
b = fe.backend.get_lr(model=m)  # 0.001
```

This method can be used with PyTorch models:
```python
m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
b = fe.backend.get_lr(model=m)  # 0.001
```


<h3>Args:</h3>

* **model** :  A neural network instance to inspect.

<h3>Returns:</h3>
    The learning rate of `model`.

<h3>Raises:</h3>

* **ValueError** :  If `model` is an unacceptable data type.

