## clip_by_value<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/clip_by_value.py/#L24-L81>View source on Github</a>
```python
clip_by_value(
	tensor: ~Tensor,
	min_value: Union[int, float, ~Tensor, NoneType]=None,
	max_value: Union[int, float, ~Tensor, NoneType]=None
)
-> ~Tensor
```
Clip a tensor such that `min_value` &lt;= tensor &lt;= `max_value`.

Given an interval, values outside the interval are clipped. If `min_value` or `max_value` is not provided then
clipping is not performed on lower or upper interval edge respectively.

This method can be used with Numpy data:
```python
n = np.array([-5, 4, 2, 0, 9, -2])
b = fe.backend.clip_by_value(n, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
b = fe.backend.clip_by_value(n, min_value=-2) # [-2, 4, 2, 0, 9, -2]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-5, 4, 2, 0, 9, -2])
b = fe.backend.clip_by_value(t, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
b = fe.backend.clip_by_value(t, min_value=-2) # [-2, 4, 2, 0, 9, -2]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-5, 4, 2, 0, 9, -2])
b = fe.backend.clip_by_value(p, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
b = fe.backend.clip_by_value(p, min_value=-2) # [-2, 4, 2, 0, 9, -2]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **min_value**: The minimum value to clip to.

* **max_value**: The maximum value to clip to. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The <code>tensor</code>, with it's values clipped.

</li></ul>

