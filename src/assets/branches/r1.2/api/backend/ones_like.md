## ones_like<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/ones_like.py/#L26-L67>View source on Github</a>
```python
ones_like(
	tensor: ~Tensor,
	dtype: Union[NoneType, str]=None
)
-> ~Tensor
```
Generate ones shaped like `tensor` with a specified `dtype`.

This method can be used with Numpy data:
```python
n = np.array([[0,1],[2,3]])
b = fe.backend.ones_like(n)  # [[1, 1], [1, 1]]
b = fe.backend.ones_like(n, dtype="float32")  # [[1.0, 1.0], [1.0, 1.0]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1],[2,3]])
b = fe.backend.ones_like(t)  # [[1, 1], [1, 1]]
b = fe.backend.ones_like(t, dtype="float32")  # [[1.0, 1.0], [1.0, 1.0]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1],[2,3]])
b = fe.backend.ones_like(p)  # [[1, 1], [1, 1]]
b = fe.backend.ones_like(p, dtype="float32")  # [[1.0, 1.0], [1.0, 1.0]]
```


<h3>Args:</h3>


* **tensor**: The tensor whose shape will be copied.

* **dtype**: The data type to be used when generating the resulting tensor. If None then the `tensor` dtype is used. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    A tensor of ones with the same shape as <code>tensor</code>.

</li></ul>

