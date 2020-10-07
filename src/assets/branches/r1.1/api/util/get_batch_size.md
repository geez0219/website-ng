## get_batch_size<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/util.py/#L720-L733>View source on Github</a>
```python
get_batch_size(
	data: Dict[str, Any]
)
-> int
```
Infer batch size from a batch dictionary. It will ignore all dictionary value with data type that
doesn't have "shape" attribute.


<h3>Args:</h3>

* **data** :  The batch dictionary.

<h3>Returns:</h3>
    batch size.

