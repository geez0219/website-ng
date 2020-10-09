## load_model<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/load_model.py/#L23-L66>View source on Github</a>
```python
load_model(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	weights_path: str,
	load_optimizer: bool=False
)
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


<h3>Args:</h3>


* **model**: A neural network instance to load.

* **weights_path**: Path to the `model` weights.

* **load_optimizer**: Whether to load optimizer. If True, then it will load <weights_opt> file in the path. 

<h3>Raises:</h3>


* **ValueError**: If `model` is an unacceptable data type.

