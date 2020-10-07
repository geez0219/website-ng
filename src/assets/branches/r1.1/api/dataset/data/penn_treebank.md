# fastestimator.dataset.data.penn_treebank<span class="tag">module</span>
---
## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/data/penn_treebank.py/#L28-L80>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None,
	seq_length: int=64
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, List[str]]
```
Load and return the Penn TreeBank dataset.


<h3>Args:</h3>

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.
* **seq_length** :  Length of data sequence.

<h3>Returns:</h3>
    (train_data, eval_data, test_data, vocab)

