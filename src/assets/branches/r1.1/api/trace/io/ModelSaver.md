## ModelSaver<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/model_saver.py/#L29-L62>View source on Github</a>
```python
ModelSaver(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	save_dir: str,
	frequency: int=1,
	max_to_keep: Union[int, NoneType]=None
)
-> None
```
Save model weights based on epoch frequency during training.


<h3>Args:</h3>

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the `model`.
* **frequency** :  Model saving frequency in epoch(s).
* **max_to_keep** :  Maximum number of latest saved files to keep. If 0 or None, all models will be saved.



