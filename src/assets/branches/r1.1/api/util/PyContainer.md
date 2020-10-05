## PyContainer
```python
PyContainer(
	data: Union[list, tuple, set, dict],
	truncate: Union[int, NoneType]=None
)
```
A class to convert python containers to a LaTeX representation.

This class is intentionally not @traceable.


#### Args:

* **data** :  The python object to be converted to LaTeX.
* **truncate** :  How many values to display before truncating with an ellipsis. This should be a positive integer or        None to disable truncation.