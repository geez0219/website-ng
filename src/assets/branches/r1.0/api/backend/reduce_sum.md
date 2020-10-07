## reduce_sum<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/backend/reduce_sum.py/#L24-L76>View source on Github</a>
```python
reduce_sum(
	tensor: ~Tensor,
	axis: Union[NoneType, int, Sequence[int]]=None,
	keepdims: bool=False
)
-> ~Tensor
```
Compute the sum along a given `axis` of a `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_sum(n)  # 36
b = fe.backend.reduce_sum(n, axis=0)  # [[6, 8], [10, 12]]
b = fe.backend.reduce_sum(n, axis=1)  # [[4, 6], [12, 14]]
b = fe.backend.reduce_sum(n, axis=[0,2])  # [14, 22]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_sum(t)  # 36
b = fe.backend.reduce_sum(t, axis=0)  # [[6, 8], [10, 12]]
b = fe.backend.reduce_sum(t, axis=1)  # [[4, 6], [12, 14]]
b = fe.backend.reduce_sum(t, axis=[0,2])  # [14, 22]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_sum(p)  # 36
b = fe.backend.reduce_sum(p, axis=0)  # [[6, 8], [10, 12]]
b = fe.backend.reduce_sum(p, axis=1)  # [[4, 6], [12, 14]]
b = fe.backend.reduce_sum(p, axis=[0,2])  # [14, 22]
```


<h3>Args:</h3>

* **tensor** :  The input value.
* **axis** :  Which axis or collection of axes to compute the sum along.
* **keepdims** :  Whether to preserve the number of dimensions during the reduction.

<h3>Returns:</h3>
    The sum of `tensor` along `axis`.

<h3>Raises:</h3>

* **ValueError** :  If `tensor` is an unacceptable data type.

