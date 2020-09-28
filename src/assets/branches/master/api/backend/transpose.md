

### transpose
```python
transpose(
	tensor: ~Tensor
)
-> ~Tensor
```
Transpose the `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(n)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(t)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
b = fe.backend.transpose(p)  # [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
```


#### Args:

* **tensor** :  The input value.

#### Returns:
    The transposed `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.