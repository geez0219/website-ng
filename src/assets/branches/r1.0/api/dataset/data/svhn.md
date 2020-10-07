# fastestimator.dataset.data.svhn<span class="tag">module</span>
---
## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/data/svhn.py/#L110-L163>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.pickle_dataset.PickleDataset, fastestimator.dataset.pickle_dataset.PickleDataset]
```
Load and return the Street View House Numbers (SVHN) dataset.


<h3>Args:</h3>

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

<h3>Returns:</h3>
    (train_data, test_data)

