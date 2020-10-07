## NumpyDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/numpy_dataset.py/#L24-L46>View source on Github</a>
```python
NumpyDataset(
	data: Dict[str, Union[numpy.ndarray, List]]
)
-> None
```
A dataset constructed from a dictionary of Numpy data or list of data.


<h3>Args:</h3>

* **data** :  A dictionary of data like {"key1" <numpy array>, "key2" [list]}.

<h3>Raises:</h3>

* **AssertionError** :  If any of the Numpy arrays or lists have differing numbers of elements.
* **ValueError** :  If any dictionary value is not instance of Numpy array or list.



