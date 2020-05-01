## PickleDataset
```python
PickleDataset(file_path:str) -> None
```
A dataset from a pickle file.    PickleDataset reads entries from pickled pandas data-frames. The root directory of the pickle file may be accessed    using dataset.parent_path. This may be useful if the file contains relative path information that you want to feed    into, say, an ImageReader Op.

#### Args:

* **file_path** :  The (absolute) path to the pickle file.    