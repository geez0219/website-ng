

### german_ner.load_data
```python
german_ner.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, Set[str], Set[str]]
```
Load and return the GermEval dataset.

Dataset from GermEval 2014 contains 31,000 sentences corresponding to over 590,000 tokens from German wikipedia
and News corpora. The sentence is encoded as one token per line with information provided in tab-seprated columns.
Sourced from https://sites.google.com/site/germeval2014ner/data


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, eval_data, train_vocab, label_vocab)