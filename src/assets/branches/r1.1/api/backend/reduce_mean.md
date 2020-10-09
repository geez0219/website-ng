## reduce_mean<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/reduce_mean.py/#L26-L84>View source on Github</a>
```python
reduce_mean(
	tensor: ~Tensor,
	axis: Union[NoneType, int, Sequence[int]]=None,
	keepdims: bool=False
)
-> ~Tensor
```
Compute the mean value along a given `axis` of a `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_mean(n)  # 4.5
b = fe.backend.reduce_mean(n, axis=0)  # [[3, 4], [5, 6]]
b = fe.backend.reduce_mean(n, axis=1)  # [[2, 3], [6, 7]]
b = fe.backend.reduce_mean(n, axis=[0,2])  # [3.5, 5.5]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_mean(t)  # 4.5
b = fe.backend.reduce_mean(t, axis=0)  # [[3, 4], [5, 6]]
b = fe.backend.reduce_mean(t, axis=1)  # [[2, 3], [3, 7]]
b = fe.backend.reduce_mean(t, axis=[0,2])  # [3.5, 5.5]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.reduce_mean(p)  # 4.5
b = fe.backend.reduce_mean(p, axis=0)  # [[3, 4], [5, 6]]
b = fe.backend.reduce_mean(p, axis=1)  # [[2, 3], [6, 7]]
b = fe.backend.reduce_mean(p, axis=[0,2])  # [3.5, 5.5]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **axis**: Which axis or collection of axes to compute the mean along.

* **keepdims**: Whether to preserve the number of dimensions during the reduction. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The mean values of <code>tensor</code> along <code>axis</code>.

</li></ul>

