## ModelSaver<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/io/model_saver.py/#L25-L44>View source on Github</a>
```python
ModelSaver(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	save_dir: str,
	frequency: int=1
)
-> None
```
Save model weights based on epoch frequency during training.


<h3>Args:</h3>

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the `model`.
* **frequency** :  Model saving frequency in epoch(s).



