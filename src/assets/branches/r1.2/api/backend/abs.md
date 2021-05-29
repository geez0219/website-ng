## abs<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/abs.py/#L24-L61>View source on Github</a>
```python
abs(
	tensor: ~Tensor
)
-> ~Tensor
```
Compute the absolute value of a tensor.

This method can be used with Numpy data:
```python
n = np.array([-2, 7, -19])
b = fe.backend.abs(n)  # [2, 7, 19]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-2, 7, -19])
b = fe.backend.abs(t)  # [2, 7, 19]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-2, 7, -19])
b = fe.backend.abs(p)  # [2, 7, 19]
```


<h3>Args:</h3>


* **tensor**: The input value. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The absolute value of <code>tensor</code>.

</li></ul>

