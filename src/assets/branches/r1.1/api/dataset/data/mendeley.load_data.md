

### mendeley.load_data
```python
mendeley.load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.labeled_dir_dataset.LabeledDirDataset, fastestimator.dataset.labeled_dir_dataset.LabeledDirDataset]
```
Load and return the Mendeley dataset.

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
Images for Classification", Mendeley Data, v2 http://dx.doi.org/10.17632/rscbjbr9sj.2

CC BY 4.0 licence:
https://creativecommons.org/licenses/by/4.0/


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, test_data)