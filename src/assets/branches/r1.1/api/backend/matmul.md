## matmul<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/matmul.py/#L24-L67>View source on Github</a>
```python
matmul(
	a: ~Tensor,
	b: ~Tensor
)
-> ~Tensor
```
Perform matrix multiplication on `a` and `b`.

This method can be used with Numpy data:
```python
a = np.array([[0,1,2],[3,4,5]])
b = np.array([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```

This method can be used with TensorFlow tensors:
```python
a = tf.constant([[0,1,2],[3,4,5]])
b = tf.constant([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```

This method can be used with PyTorch tensors:
```python
a = torch.tensor([[0,1,2],[3,4,5]])
b = torch.tensor([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```


<h3>Args:</h3>


* **a**: The first matrix.

* **b**: The second matrix. 

<h3>Raises:</h3>


* **ValueError**: If either `a` or `b` are unacceptable or non-matching data types.

<h3>Returns:</h3>

<ul class="return-block"><li>    The matrix multiplication result of a * b.

</li></ul>

