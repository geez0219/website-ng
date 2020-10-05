

### food101.load_data
```python
food101.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.csv_dataset.CSVDataset, fastestimator.dataset.csv_dataset.CSVDataset]
```
Load and return the Food-101 dataset.

Food-101 dataset is a collection of images from 101 food categories.
Sourced from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, test_data)