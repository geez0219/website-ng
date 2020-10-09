## reduce_min<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/reduce_min.py/#L26-L82>View source on Github</a>
```python
reduce_min(
	tensor: ~Tensor,
	axis: Union[NoneType, int, Sequence[int]]=None,
	keepdims: bool=False
)
-> ~Tensor
```
Compute the min value along a given `axis` of a `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_min(n)  # 1
b = fe.backend.reduce_min(n, axis=0)  # [[1, 2], [3, 4]]
b = fe.backend.reduce_min(n, axis=1)  # [[1, 2], [5, 6]]
b = fe.backend.reduce_min(n, axis=[0,2])  # [1, 3]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_min(t)  # 1
b = fe.backend.reduce_min(t, axis=0)  # [[1, 2], [3, 4]]
b = fe.backend.reduce_min(t, axis=1)  # [[1, 2], [5, 6]]
b = fe.backend.reduce_min(t, axis=[0,2])  # [1, 3]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_min(p)  # 1
b = fe.backend.reduce_min(p, axis=0)  # [[1, 2], [3, 4]]
b = fe.backend.reduce_min(p, axis=1)  # [[1, 2], [5, 6]]
b = fe.backend.reduce_min(p, axis=[0,2])  # [1, 3]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **axis**: Which axis or collection of axes to compute the min along.

* **keepdims**: Whether to preserve the number of dimensions during the reduction. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The min values of <code>tensor</code> along <code>axis</code>.

</li></ul>

