

### maximum
```python
maximum(
	tensor1: ~Tensor,
	tensor2: ~Tensor
)
-> ~Tensor
```
Get the maximum of the given `tensors`.

This method can be used with Numpy data:
```python
n1 = np.array([[2, 7, 6]])
n2 = np.array([[2, 7, 5]])
res = fe.backend.maximum(n1, n2) # [[2, 7, 6]]
```

This method can be used with TensorFlow tensors:
```python
t1 = tf.constant([[2, 7, 6]])
t2 = tf.constant([[2, 7, 5]])
res = fe.backend.maximum(t1, t2) # [[2, 7, 6]]
```

This method can be used with PyTorch tensors:
```python
p1 = torch.tensor([[2, 7, 6]])
p2 = torch.tensor([[2, 7, 5]])
res = fe.backend.maximum(p1, p2) # [[2, 7, 6]]
```


#### Args:

* **tensor1** :  First tensor.
* **tensor2** :  Second tensor.

#### Returns:
    The maximum of two `tensors`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.