## sign<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/sign.py/#L24-L61>View source on Github</a>
```python
sign(
	tensor: ~Tensor
)
-> ~Tensor
```
Compute the sign of a tensor.

This method can be used with Numpy data:
```python
n = np.array([-2, 7, -19])
b = fe.backend.sign(n)  # [-1, 1, -1]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-2, 7, -19])
b = fe.backend.sign(t)  # [-1, 1, -1]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-2, 7, -19])
b = fe.backend.sign(p)  # [-1, 1, -1]
```


<h3>Args:</h3>


* **tensor**: The input value. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The sign of each value of the <code>tensor</code>.

</li></ul>

