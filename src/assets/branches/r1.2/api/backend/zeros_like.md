## zeros_like<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/zeros_like.py/#L26-L67>View source on Github</a>
```python
zeros_like(
	tensor: ~Tensor,
	dtype: Union[NoneType, str]=None
)
-> ~Tensor
```
Generate zeros shaped like `tensor` with a specified `dtype`.

This method can be used with Numpy data:
```python
n = np.array([[0,1],[2,3]])
b = fe.backend.zeros_like(n)  # [[0, 0], [0, 0]]
b = fe.backend.zeros_like(n, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1],[2,3]])
b = fe.backend.zeros_like(t)  # [[0, 0], [0, 0]]
b = fe.backend.zeros_like(t, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1],[2,3]])
b = fe.backend.zeros_like(p)  # [[0, 0], [0, 0]]
b = fe.backend.zeros_like(p, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
```


<h3>Args:</h3>


* **tensor**: The tensor whose shape will be copied.

* **dtype**: The data type to be used when generating the resulting tensor. If None then the `tensor` dtype is used. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    A tensor of zeros with the same shape as <code>tensor</code>.

</li></ul>

