## random_uniform_like<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/random_uniform_like.py/#L26-L71>View source on Github</a>
```python
random_uniform_like(
	tensor: ~Tensor,
	minval: float=0.0,
	maxval: float=1.0,
	dtype: Union[NoneType, str]='float32'
)
-> ~Tensor
```
Generate noise shaped like `tensor` from a random normal distribution with a given `mean` and `std`.

This method can be used with Numpy data:
```python
n = np.array([[0,1],[2,3]])
b = fe.backend.random_uniform_like(n)  # [[0.62, 0.49], [0.88, 0.37]]
b = fe.backend.random_uniform_like(n, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1],[2,3]])
b = fe.backend.random_uniform_like(t)  # [[0.62, 0.49], [0.88, 0.37]]
b = fe.backend.random_uniform_like(t, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1],[2,3]])
b = fe.backend.random_uniform_like(p)  # [[0.62, 0.49], [0.88, 0.37]]
b = fe.backend.random_uniform_like(P, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
```


<h3>Args:</h3>


* **tensor**: The tensor whose shape will be copied.

* **minval**: The minimum bound of the uniform distribution.

* **maxval**: The maximum bound of the uniform distribution.

* **dtype**: The data type to be used when generating the resulting tensor. This should be one of the floating point types. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    A tensor of random uniform noise with the same shape as <code>tensor</code>.

</li></ul>

