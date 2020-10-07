## check_nan<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/check_nan.py/#L22-L54>View source on Github</a>
```python
check_nan(
	val: Union[int, float, numpy.ndarray, tensorflow.python.framework.ops.Tensor, torch.Tensor]
)
-> bool
```
Checks if the input contains NaN values.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, np.NaN]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.check_nan(n)  # True
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[np.NaN, 6.0], [7.0, 8.0]]])
b = fe.backend.check_nan(n)  # True
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [np.NaN, 8.0]]])
b = fe.backend.check_nan(n)  # True
```


<h3>Args:</h3>

* **val** :  The input value.

<h3>Returns:</h3>
    True iff `val` contains NaN

