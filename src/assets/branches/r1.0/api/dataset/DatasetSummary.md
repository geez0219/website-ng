## DatasetSummary
```python
DatasetSummary(num_instances:int, keys:Dict[str, dataset.dataset.KeySummary], num_classes:Union[int, NoneType]=None, class_key_mapping:Union[Dict[str, Any], NoneType]=None, class_key:Union[str, NoneType]=None)
```
This class contains information summarizing a dataset object.

#### Args:

* **num_instances** :  The number of data instances within the dataset (influences the size of an epoch).
* **num_classes** :  How many different classes are present.
* **keys** :  What keys does the dataset provide, along with summary information about each key.
* **class_key** :  Which key corresponds to class information (if known).
* **class_key_mapping** :  A mapping of the original class string values to the values which are output to the pipeline.    