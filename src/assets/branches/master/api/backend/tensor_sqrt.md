

### tensor_sqrt
```python
tensor_sqrt(tensor:~Tensor) -> ~Tensor
```
Computes element-wise square root of tensor elements.

This method can be used with Numpy data:
```python
n = np.array([[1, 4, 6], [4, 9, 16]])
b = fe.backend.tensor_sqrt(n)  # [[1.0, 2.0, 2.44948974], [2.0, 3.0, 4.0]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[1, 4, 6], [4, 9, 16]], dtype=tf.float32)
b = fe.backend.tensor_sqrt(t)  # [[1.0, 2.0, 2.4494898], [2.0, 3.0, 4.0]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[1, 4, 6], [4, 9, 16]], dtype=torch.float32)
b = fe.backend.tensor_sqrt(p)  # [[1.0, 2.0, 2.4495], [2.0, 3.0, 4.0]]
```


#### Args:

* **tensor** :  The input tensor.

#### Returns:
    The `tensor` that contains square root of input values.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.