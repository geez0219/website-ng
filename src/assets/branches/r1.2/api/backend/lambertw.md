## lambertw<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/lambertw.py/#L27-L69>View source on Github</a>
```python
lambertw(
	tensor: ~Tensor
)
-> ~Tensor
```
Compute the k=0 branch of the Lambert W function.

See https://en.wikipedia.org/wiki/Lambert_W_function for details. Only valid for inputs &gt;= -1/e (approx -0.368). We
do not check this for the sake of speed, but if an input is out of domain the return value may be random /
inconsistent or even NaN.

This method can be used with Numpy data:
```python
n = np.array([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
b = fe.backend.lambertw(n)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
b = fe.backend.lambertw(t)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([-1.0/math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
b = fe.backend.lambertw(p)  # [-1, -0.654, -0.560, -0.259, 0, 0.108, 0.132, 1, 1.327, 2.718, 3.386]
```


<h3>Args:</h3>


* **tensor**: The input value. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The lambertw function evaluated at <code>tensor</code>.

</li></ul>

