## pow<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/pow.py/#L24-L62>View source on Github</a>
```python
pow(
	tensor: ~Tensor,
	power: Union[int, float, ~Tensor]
)
-> ~Tensor
```
Raise a `tensor` to a given `power`.

This method can be used with Numpy data:
```python
n = np.array([-2, 7, -19])
b = fe.backend.pow(n, 2)  # [4, 49, 361]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-2, 7, -19])
b = fe.backend.pow(t, 2)  # [4, 49, 361]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-2, 7, -19])
b = fe.backend.pow(p, 2)  # [4, 49, 361]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **power**: The exponent to raise `tensor` by. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The exponentiated <code>tensor</code>.

</li></ul>

