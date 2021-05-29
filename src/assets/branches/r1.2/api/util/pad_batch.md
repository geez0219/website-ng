## pad_batch<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/util.py/#L512-L539>View source on Github</a>
```python
pad_batch(
	batch: List[MutableMapping[str, numpy.ndarray]],
	pad_value: Union[float, int]
)
-> None
```
A function to pad a batch of data in-place by appending to the ends of the tensors. Tensor type needs to be
numpy array otherwise would get ignored. (tf.Tensor and torch.Tensor will cause error)

```python
data = [{"x": np.ones((2, 2)), "y": 8}, {"x": np.ones((3, 1)), "y": 4}]
fe.util.pad_batch(data, pad_value=0)
print(data)  # [{'x': [[1., 1.], [1., 1.], [0., 0.]], 'y': 8}, {'x': [[1., 0.], [1., 0.], [1., 0.]]), 'y': 4}]
```


<h3>Args:</h3>


* **batch**: A list of data to be padded.

* **pad_value**: The value to pad with. 

<h3>Raises:</h3>


* **AssertionError**: If the data within the batch do not have matching rank, or have different keys

