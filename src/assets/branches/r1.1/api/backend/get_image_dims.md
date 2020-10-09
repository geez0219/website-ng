## get_image_dims<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/get_image_dims.py/#L24-L61>View source on Github</a>
```python
get_image_dims(
	tensor: ~Tensor
)
-> ~Tensor
```
Get the `tensor` height, width and channels.

This method can be used with Numpy data:
```python
n = np.random.random((2, 12, 12, 3))
b = fe.backend.get_image_dims(n)  # (3, 12, 12)
```

This method can be used with TensorFlow tensors:
```python
t = tf.random.uniform((2, 12, 12, 3))
b = fe.backend.get_image_dims(t)  # (3, 12, 12)
```

This method can be used with PyTorch tensors:
```python
p = torch.rand((2, 3, 12, 12))
b = fe.backend.get_image_dims(p)  # (3, 12, 12)
```


<h3>Args:</h3>


* **tensor**: The input tensor. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    Channels, height and width of the <code>tensor</code>.

</li></ul>

