

### load_dict
```python
load_dict(dict_path, array_key=False)
```


#### Args:

* **dict_path** :  The path to a json dictionary
* **array_key** :  If true the parser will consider the first element in a sublist at the key {_[K,...,V],...}.
* **Otherwise it will parse as {K** : V,...} or {K[...,V],...}

#### Returns:
    A dictionary corresponding to the info from the file. If the file was formatted with arrays as the values for a    key, the last element of the array is used as the value for the key in the parsed dictionary