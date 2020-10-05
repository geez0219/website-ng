

### matmul
```python
matmul(
	a: ~Tensor,
	b: ~Tensor
)
-> ~Tensor
```
Perform matrix multiplication on `a` and `b`.

This method can be used with Numpy data:
```python
a = np.array([[0,1,2],[3,4,5]])
b = np.array([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```

This method can be used with TensorFlow tensors:
```python
a = tf.constant([[0,1,2],[3,4,5]])
b = tf.constant([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```

This method can be used with PyTorch tensors:
```python
a = torch.tensor([[0,1,2],[3,4,5]])
b = torch.tensor([[1],[2],[3]])
c = fe.backend.matmul(a, b)  # [[8], [26]]
```


#### Args:

* **a** :  The first matrix.
* **b** :  The second matrix.

#### Returns:
    The matrix multiplication result of a * b.

#### Raises:

* **ValueError** :  If either `a` or `b` are unacceptable or non-matching data types.