## ModelSaver
```python
ModelSaver(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], save_dir:str, frequency:int=1) -> None
```
Save model weights based on epoch frequency during training.

#### Args:

* **model** :  A model instance compiled with fe.build.
* **save_dir** :  Folder path into which to save the `model`.
* **frequency** :  Model saving frequency in epoch(s).    