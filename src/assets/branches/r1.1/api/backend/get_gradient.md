## get_gradient<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/get_gradient.py/#L25-L100>View source on Github</a>
```python
get_gradient(
	target: ~Tensor,
	sources: Union[Iterable[~Tensor], ~Tensor],
	higher_order: bool=False,
	tape: Union[tensorflow.python.eager.backprop.GradientTape, NoneType]=None,
	retain_graph: bool=True
)
-> Union[Iterable[~Tensor], ~Tensor]
```
Calculate gradients of a target w.r.t sources.

This method can be used with TensorFlow tensors:
```python
x = tf.Variable([1.0, 2.0, 3.0])
with tf.GradientTape(persistent=True) as tape:
    y = x * x

    b = fe.backend.get_gradient(target=y, sources=x, tape=tape)  # [2.0, 4.0, 6.0]
    b = fe.backend.get_gradient(target=b, sources=x, tape=tape)  # None

    b = fe.backend.get_gradient(target=y, sources=x, tape=tape, higher_order=True)  # [2.0, 4.0, 6.0]
    b = fe.backend.get_gradient(target=b, sources=x, tape=tape)  # [2.0, 2.0, 2.0]
```

This method can be used with PyTorch tensors:
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x

b = fe.backend.get_gradient(target=y, sources=x)  # [2.0, 4.0, 6.0]
b = fe.backend.get_gradient(target=b, sources=x)  # Error - b does not have a backwards function

b = fe.backend.get_gradient(target=y, sources=x, higher_order=True)  # [2.0, 4.0, 6.0]
b = fe.backend.get_gradient(target=b, sources=x)  # [2.0, 2.0, 2.0]
```


<h3>Args:</h3>

* **target** :  The target (final) tensor.
* **sources** :  A sequence of source (initial) tensors.
* **higher_order** :  Whether the gradient will be used for higher order gradients.
* **tape** :  TensorFlow gradient tape. Only needed when using the TensorFlow backend.
* **retain_graph** :  Whether to retain PyTorch graph. Only valid when using the PyTorch backend.

<h3>Returns:</h3>
    Gradient(s) of the `target` with respect to the `sources`.

<h3>Raises:</h3>

* **ValueError** :  If `target` is an unacceptable data type.

