

### usps.load_data
```python
usps.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.labeled_dir_dataset.LabeledDirDataset, fastestimator.dataset.labeled_dir_dataset.LabeledDirDataset]
```
Load and return the USPS dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, test_data)