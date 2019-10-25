

### load_data
```python
load_data(path=None)
```
Download the MNIST dataset to local storage, if not already downloaded. This will generate 2 csv files(train, eval), which contain all the path information.

#### Args:

* **path (str, optional)** :  The path to store the MNIST data. When `path` is not provided, will save at        `fastestimator_data` under user's home directory.

#### Returns:

* **(tuple)** :  tuple containing
* **train_csv (str)** :  Path to train csv file.
* **eval_csv (str)** :  Path to test csv file.
* **path (str)** :  Path to data root directory.