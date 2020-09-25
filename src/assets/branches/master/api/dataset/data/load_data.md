

### load_data
```python
load_data(root_dir:Union[str, NoneType]=None, seq_length:int=100) -> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, List[str]]
```
Load and return the Shakespeare dataset.

Shakespeare dataset is a collection of texts written by Shakespeare.
Sourced from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.
* **seq_length** :  Length of data sequence.

#### Returns:
    (train_data, vocab)