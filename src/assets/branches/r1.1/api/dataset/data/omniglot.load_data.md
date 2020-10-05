

### omniglot.load_data
```python
omniglot.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.siamese_dir_dataset.SiameseDirDataset, fastestimator.dataset.siamese_dir_dataset.SiameseDirDataset]
```
Load and return the Omniglot dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, eval_data)