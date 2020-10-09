## concat<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/backend/concat.py/#L24-L67>View source on Github</a>
```python
concat(
	tensors: List[~Tensor],
	axis: int=0
)
-> Union[~Tensor, NoneType]
```
Concatenate a list of `tensors` along a given `axis`.

This method can be used with Numpy data:
```python
n = [np.array([[0, 1]]), np.array([[2, 3]]), np.array([[4, 5]])]
b = fe.backend.concat(n, axis=0)  # [[0, 1], [2, 3], [4, 5]]
b = fe.backend.concat(n, axis=1)  # [[0, 1, 2, 3, 4, 5]]
```

This method can be used with TensorFlow tensors:
```python
t = [tf.constant([[0, 1]]), tf.constant([[2, 3]]), tf.constant([[4, 5]])]
b = fe.backend.concat(t, axis=0)  # [[0, 1], [2, 3], [4, 5]]
b = fe.backend.concat(t, axis=1)  # [[0, 1, 2, 3, 4, 5]]
```

This method can be used with PyTorch tensors:
```python
p = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])]
b = fe.backend.concat(p, axis=0)  # [[0, 1], [2, 3], [4, 5]]
b = fe.backend.concat(p, axis=1)  # [[0, 1, 2, 3, 4, 5]]
```


<h3>Args:</h3>


* **tensors**: A list of tensors to be concatenated.

* **axis**: The axis along which to concatenate the input. 

<h3>Raises:</h3>


* **ValueError**: If `tensors` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    A concatenated representation of the <code>tensors</code>, or None if the list of <code>tensors</code> was empty.

</li></ul>

