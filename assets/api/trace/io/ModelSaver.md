## ModelSaver
```python
ModelSaver(model_name, save_dir, save_best=False, save_best_mode='min', save_freq=1)
```
Save trained model in hdf5 format.

#### Args:

* **model_name (str)** :  Name of FE model.
* **save_dir (str)** :  Directory to save the trained models.
* **save_best (bool, str, optional)** :  Best model saving monitor name. If True, the model loss is used. Defaults to        False.
* **save_best_mode (str, optional)** :  Can be `'min'`, `'max'`, or `'auto'`. Defaults to 'min'.
* **save_freq (int, optional)** :  Number of epochs to save models. Cannot be used with `save_best_only=True`. Defaults        to 1.