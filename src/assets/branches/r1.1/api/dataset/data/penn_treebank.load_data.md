

### penn_treebank.load_data
```python
penn_treebank.load_data(
	root_dir: Union[str, NoneType]=None,
	seq_length: int=64
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, List[str]]
```
Load and return the Penn TreeBank dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.
* **seq_length** :  Length of data sequence.

#### Returns:
    (train_data, eval_data, test_data, vocab)