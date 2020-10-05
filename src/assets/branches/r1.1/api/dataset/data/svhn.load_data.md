

### svhn.load_data
```python
svhn.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.pickle_dataset.PickleDataset, fastestimator.dataset.pickle_dataset.PickleDataset]
```
Load and return the Street View House Numbers (SVHN) dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, test_data)