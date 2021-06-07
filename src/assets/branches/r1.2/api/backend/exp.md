## exp<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/exp.py/#L24-L61>View source on Github</a>
```python
exp(
	tensor: ~Tensor
)
-> ~Tensor
```
Compute e^Tensor.

This method can be used with Numpy data:
```python
n = np.array([-2, 2, 1])
b = fe.backend.exp(n)  # [0.1353, 7.3891, 2.7183]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-2.0, 2, 1])
b = fe.backend.exp(t)  # [0.1353, 7.3891, 2.7183]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-2.0, 2, 1])
b = fe.backend.exp(p)  # [0.1353, 7.3891, 2.7183]
```


<h3>Args:</h3>


* **tensor**: The input value. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The exponentiated <code>tensor</code>.

</li></ul>

