# fastestimator.dataset.data.food101<span class="tag">module</span>
---
## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/data/food101.py/#L46-L96>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.csv_dataset.CSVDataset, fastestimator.dataset.csv_dataset.CSVDataset]
```
Load and return the Food-101 dataset.

Food-101 dataset is a collection of images from 101 food categories.
Sourced from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz


<h3>Args:</h3>

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

<h3>Returns:</h3>
    (train_data, test_data)

