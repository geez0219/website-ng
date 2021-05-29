## transpose<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/transpose.py/#L24-L61>View source on Github</a>
```python
transpose(
	tensor: ~Tensor
)
-> ~Tensor
```
Transpose the `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(n)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(t)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(p)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```


<h3>Args:</h3>


* **tensor**: The input value. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The transposed <code>tensor</code>.

</li></ul>

