

### nih_chestxray.load_data
```python
nih_chestxray.load_data(
	root_dir: Union[str, NoneType]=None
)
-> fastestimator.dataset.dir_dataset.DirDataset
```
Load and return the NIH Chest X-ray dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    train_data