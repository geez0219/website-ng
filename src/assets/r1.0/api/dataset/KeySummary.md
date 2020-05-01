## KeySummary
```python
KeySummary(dtype:str, num_unique_values:Union[int, NoneType]=None, shape:List[Union[int, NoneType]]=()) -> None
```
A summary of the dataset attributes corresponding to a particular key.

#### Args:

* **num_unique_values** :  The number of unique values corresponding to a particular key (if known).
* **shape** :  The shape of the vectors corresponding to the key. None is used in a list to indicate that a dimension is            ragged.
* **dtype** :  The data type of instances corresponding to the given key.    