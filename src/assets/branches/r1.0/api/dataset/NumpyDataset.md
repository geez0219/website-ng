## NumpyDataset
```python
NumpyDataset(data:Dict[str, numpy.ndarray]) -> None
```
A dataset constructed from a dictionary of Numpy data.


#### Args:

* **data** :  A dictionary of data like {"key" &lt;numpy array&gt;}.

#### Raises:

* **AssertionError** :  If any of the Numpy arrays have differing numbers of elements.