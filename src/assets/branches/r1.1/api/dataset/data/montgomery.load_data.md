

### montgomery.load_data
```python
montgomery.load_data(
	root_dir: Union[str, NoneType]=None
)
-> fastestimator.dataset.csv_dataset.CSVDataset
```
Load and return the montgomery dataset.

Sourced from http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip. This method will download the data
    to local storage if the data has not been previously downloaded.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

#### Returns:
    train_data