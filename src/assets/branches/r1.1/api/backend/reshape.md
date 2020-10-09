## reshape<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/reshape.py/#L24-L72>View source on Github</a>
```python
reshape(
	tensor: ~Tensor,
	shape: List[int]
)
-> ~Tensor
```
Reshape a `tensor` to conform to a given shape.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reshape(n, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
b = fe.backend.reshape(n, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
b = fe.backend.reshape(n, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
b = fe.backend.reshape(n, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reshape(t, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
b = fe.backend.reshape(t, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
b = fe.backend.reshape(t, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
b = fe.backend.reshape(t, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reshape(p, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
b = fe.backend.reshape(p, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
b = fe.backend.reshape(p, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
b = fe.backend.reshape(p, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **shape**: The new shape of the tensor. At most one value may be -1 which indicates that whatever values are left should be packed into that axis. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The reshaped <code>tensor</code>.

</li></ul>

