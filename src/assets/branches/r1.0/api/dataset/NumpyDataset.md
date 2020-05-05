## NumpyDataset
```python
NumpyDataset(data:Dict[str, numpy.ndarray]) -> None
```
A dataset constructed from a dictionary of Numpy data.

#### Args:

* **data** :  A dictionary of data like {"key" <numpy array>}.

#### Raises:

* **AssertionError** :  If any of the Numpy arrays have differing numbers of elements.    