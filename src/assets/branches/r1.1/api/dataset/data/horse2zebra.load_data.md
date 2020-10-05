

### horse2zebra.load_data
```python
horse2zebra.load_data(
	batch_size: int,
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.batch_dataset.BatchDataset, fastestimator.dataset.batch_dataset.BatchDataset]
```
Load and return the horse2zebra dataset.

Sourced from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip. This method will
    download the data to local storage if the data has not been previously downloaded.


#### Args:

* **batch_size** :  The desired batch size.
* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    (train_data, eval_data)