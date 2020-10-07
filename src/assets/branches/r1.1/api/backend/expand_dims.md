## expand_dims<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/expand_dims.py/#L24-L65>View source on Github</a>
```python
expand_dims(
	tensor: ~Tensor,
	axis: int=1
)
-> ~Tensor
```
Create a new dimension in `tensor` along a given `axis`.

This method can be used with Numpy data:
```python
n = np.array([2,7,5])
b = fe.backend.expand_dims(n, axis=0)  # [[2, 5, 7]]
b = fe.backend.expand_dims(n, axis=1)  # [[2], [5], [7]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([2,7,5])
b = fe.backend.expand_dims(t, axis=0)  # [[2, 5, 7]]
b = fe.backend.expand_dims(t, axis=1)  # [[2], [5], [7]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([2,7,5])
b = fe.backend.expand_dims(p, axis=0)  # [[2, 5, 7]]
b = fe.backend.expand_dims(p, axis=1)  # [[2], [5], [7]]
```


<h3>Args:</h3>

* **tensor** :  The input to be modified, having n dimensions.
* **axis** :  Which axis should the new axis be inserted along. Must be in the range [-n-1, n].

<h3>Returns:</h3>
    A concatenated representation of the `tensors`, or None if the list of `tensors` was empty.

<h3>Raises:</h3>

* **ValueError** :  If `tensor` is an unacceptable data type.

