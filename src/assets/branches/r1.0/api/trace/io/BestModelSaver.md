## BestModelSaver<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/io/best_model_saver.py/#L26-L79>View source on Github</a>
```python
BestModelSaver(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	save_dir: str,
	metric: Union[str, NoneType]=None,
	save_best_mode: str='min'
)
-> None
```
Save the weights of best model based on a given evaluation metric.


<h3>Args:</h3>

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the model.
* **metric** :  Eval metric name to monitor. If None, the model's loss will be used.
* **save_best_mode** :  Can be 'min' or 'max'.

<h3>Raises:</h3>

* **AssertionError** :  If a `metric` is not provided and it cannot be inferred from the `model`.
* **ValueError** :  If `save_best_mode` is an unacceptable string.



