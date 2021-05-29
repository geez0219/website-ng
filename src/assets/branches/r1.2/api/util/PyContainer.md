## PyContainer<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/latex_util.py/#L39-L65>View source on Github</a>
```python
PyContainer(
	data: Union[list, tuple, set, dict],
	truncate: Union[int, NoneType]=None
)
```
A class to convert python containers to a LaTeX representation.

This class is intentionally not @traceable.


<h3>Args:</h3>


* **data**: The python object to be converted to LaTeX.

* **truncate**: How many values to display before truncating with an ellipsis. This should be a positive integer or None to disable truncation.

