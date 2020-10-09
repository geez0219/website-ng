## pad_batch<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L386-L408>View source on Github</a>
```python
pad_batch(
	batch: List[MutableMapping[str, Any]],
	pad_value: Union[float, int]
)
-> None
```
A function to pad a batch of data in-place by appending to the ends of the tensors.

```python
data = [{"x": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
fe.util.pad_batch(data, pad_value=0)
print(data)  # [{'x': [[1., 1.], [1., 1.],[0., 0.]], 'y': 8}, {'x': [[1., 0.], [1., 0.], [1., 0.]]), 'y': 4}]
```


<h3>Args:</h3>


* **batch**: A list of data to be padded.

* **pad_value**: The value to pad with. 

<h3>Raises:</h3>


* **AssertionError**: If the data within the batch do not have matching ranks.

