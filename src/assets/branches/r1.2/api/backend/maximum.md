## maximum<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/maximum.py/#L24-L65>View source on Github</a>
```python
maximum(
	tensor1: ~Tensor,
	tensor2: ~Tensor
)
-> ~Tensor
```
Get the maximum of the given `tensors`.

This method can be used with Numpy data:
```python
n1 = np.array([[2, 7, 6]])
n2 = np.array([[2, 7, 5]])
res = fe.backend.maximum(n1, n2) # [[2, 7, 6]]
```

This method can be used with TensorFlow tensors:
```python
t1 = tf.constant([[2, 7, 6]])
t2 = tf.constant([[2, 7, 5]])
res = fe.backend.maximum(t1, t2) # [[2, 7, 6]]
```

This method can be used with PyTorch tensors:
```python
p1 = torch.tensor([[2, 7, 6]])
p2 = torch.tensor([[2, 7, 5]])
res = fe.backend.maximum(p1, p2) # [[2, 7, 6]]
```


<h3>Args:</h3>


* **tensor1**: First tensor.

* **tensor2**: Second tensor. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The maximum of two <code>tensors</code>.

</li></ul>

