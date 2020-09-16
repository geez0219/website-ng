## ModelSaver
```python
ModelSaver(*args, **kwargs)
```
Save model weights based on epoch frequency during training.


#### Args:

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the `model`.
* **frequency** :  Model saving frequency in epoch(s).
* **max_to_keep** :  Maximum number of latest saved files to keep. If 0 or None, all models will be saved.