## roll<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/roll.py/#L24-L76>View source on Github</a>
```python
roll(
	tensor: ~Tensor,
	shift: Union[int, List[int]],
	axis: Union[int, List[int]]
)
-> ~Tensor
```
Roll a `tensor` elements along a given axis.

The elements are shifted forward or reverse direction by the offset of `shift`. Overflown elements beyond the last
position will be re-introduced at the first position.

This method can be used with Numpy data:
```python
n = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
b = fe.backend.roll(n, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
b = fe.backend.roll(n, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
b = fe.backend.roll(n, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
b = fe.backend.roll(n, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
b = fe.backend.roll(t, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
b = fe.backend.roll(t, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
b = fe.backend.roll(t, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
b = fe.backend.roll(t, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
b = fe.backend.roll(p, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
b = fe.backend.roll(p, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
b = fe.backend.roll(p, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
b = fe.backend.roll(p, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
```


<h3>Args:</h3>


* **tensor**: The input value.

* **shift**: The number of places by which the elements need to be shifted. If shift is a list, axis must be a list of same size.

* **axis**: axis along which elements will be rolled. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The rolled <code>tensor</code>.

</li></ul>

