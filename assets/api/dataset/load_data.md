

### load_data
```python
load_data(path=None)
```
Download the MNIST dataset to local storage, if not already downloaded. This will generate 2 csv files(train, eval), which contain all the path information.

#### Args:

* **path (str, optional)** :  The path to store the MNIST data. When `path` is not provided, will save at        `fastestimator_data` under user's home directory.

#### Returns:

* **tuple** :  (train_csv, eval_csv, path) tuple, where    
 * **train_csv** (str) -- Path to train csv file, containing the following columns :     
 * x (str) :  Image directory relative to the returned path.
 * y (int) :  Label indicating the number shown in the image.        * **eval_csv** (str) -- Path to test csv file, containing the same columns as train_csv.    * **path** (str) -- Path to data directory.