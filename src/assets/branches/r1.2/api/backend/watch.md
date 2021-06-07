## watch<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/watch.py/#L23-L59>View source on Github</a>
```python
watch(
	tensor: ~Tensor,
	tape: Union[tensorflow.python.eager.backprop.GradientTape, NoneType]=None
)
-> ~Tensor
```
Monitor the given `tensor` for later gradient computations.

This method can be used with TensorFlow tensors:
```python
x = tf.ones((3,28,28,1))
with tf.GradientTape(persistent=True) as tape:
    x = fe.backend.watch(x, tape=tape)
```

This method can be used with PyTorch tensors:
```python
x = torch.ones((3,1,28,28))  # x.requires_grad == False
x = fe.backend.watch(x)  # x.requires_grad == True
```


<h3>Args:</h3>


* **tensor**: The tensor to be monitored.

* **tape**: A TensorFlow GradientTape which will be used to record gradients (iff using TensorFlow for the backend). 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The <code>tensor</code> or a copy of the <code>tensor</code> which is being tracked for gradient computations. This value is only
    needed if using PyTorch as the backend.

</li></ul>

