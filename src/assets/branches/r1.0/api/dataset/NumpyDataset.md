## NumpyDataset
```python
NumpyDataset(
	data: Dict[str, Union[numpy.ndarray, List]]
)
-> None
```
A dataset constructed from a dictionary of Numpy data or list of data.


#### Args:

* **data** :  A dictionary of data like {"key1" <numpy array>, "key2" [list]}.

#### Raises:

* **AssertionError** :  If any of the Numpy arrays or lists have differing numbers of elements.
* **ValueError** :  If any dictionary value is not instance of Numpy array or list.