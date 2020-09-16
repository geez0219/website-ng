

### tensor_round
```python
tensor_round(tensor:~Tensor) -> ~Tensor
```
Element-wise rounds the values of the `tensor` to nearest integer.

This method can be used with Numpy data:
```python
n = np.array([[1.25, 4.5, 6], [4, 9.11, 16]])
b = fe.backend.tensor_round(n)  # [[1, 4, 6], [4, 9, 16]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[1.25, 4.5, 6], [4, 9.11, 16.9]])
b = fe.backend.tensor_round(t)  # [[1, 4, 6], [4, 9, 17]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[1.25, 4.5, 6], [4, 9.11, 16]])
b = fe.backend.tensor_round(p)  # [[1, 4, 6], [4, 9, 16]]
```


#### Args:

* **tensor** :  The input tensor.

#### Returns:
    The rounded `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.