## reduce_max<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/reduce_max.py/#L26-L82>View source on Github</a>
```python
reduce_max(
	tensor: ~Tensor,
	axis: Union[NoneType, int, Sequence[int]]=None,
	keepdims: bool=False
)
-> ~Tensor
```
Compute the maximum value along a given `axis` of a `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = fe.backend.reduce_max(n)  # 8
b = fe.backend.reduce_max(n, axis=0)  # [[5, 6], [7, 8]]
b = fe.backend.reduce_max(n, axis=1)  # [[3, 4], [7, 8]]
b = fe.backend.reduce_max(n, axis=[0,2])  # [6, 8]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = fe.backend.reduce_max(t)  # 8
b = fe.backend.reduce_max(t, axis=0)  # [[5, 6], [7, 8]]
b = fe.backend.reduce_max(t, axis=1)  # [[3, 4], [7, 8]]
b = fe.backend.reduce_max(t, axis=[0,2])  # [6, 8]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = fe.backend.reduce_max(p)  # 8
b = fe.backend.reduce_max(p, axis=0)  # [[5, 6], [7, 8]]
b = fe.backend.reduce_max(p, axis=1)  # [[3, 4], [7, 8]]
b = fe.backend.reduce_max(p, axis=[0,2])  # [6, 8]
```


<h3>Args:</h3>

* **tensor** :  The input value.
* **axis** :  Which axis or collection of axes to compute the maximum along.
* **keepdims** :  Whether to preserve the number of dimensions during the reduction.

<h3>Returns:</h3>
    The maximum values of `tensor` along `axis`.

<h3>Raises:</h3>

* **ValueError** :  If `tensor` is an unacceptable data type.

