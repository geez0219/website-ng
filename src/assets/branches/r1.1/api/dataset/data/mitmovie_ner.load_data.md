

### mitmovie_ner.load_data
```python
mitmovie_ner.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, Set[str], Set[str]]
```
Load and return the MIT Movie dataset.

MIT Movies dataset is a semantically tagged training and test corpus in BIO format. The sentence is encoded as one
token per line with information provided in tab-seprated columns.
Sourced from https://groups.csail.mit.edu/sls/downloads/movie/


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, eval_data, train_vocab, label_vocab)