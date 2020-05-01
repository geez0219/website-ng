## LabeledDirDataset
```python
LabeledDirDataset(root_dir:str, data_key:str='x', label_key:str='y', label_mapping:Union[Dict[str, Any], NoneType]=None, file_extension:Union[str, NoneType]=None) -> None
```
A dataset which reads files from a folder hierarchy like root/class(/es)/data.file.

#### Args:

* **root_dir** :  The path to the directory containing data sorted by folders.
* **data_key** :  What key to assign to the data values in the data dictionary.
* **label_key** :  What key to assign to the label values in the data dictionary.
* **label_mapping** :  A dictionary defining the mapping to use. If not provided will map classes to int labels.
* **file_extension** :  If provided then only files ending with the file_extension will be included.    

### summary
```python
summary(self) -> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

#### Returns:
            A summary representation of this dataset.        