

### get_image_dims
```python
get_image_dims(tensor:~Tensor) -> ~Tensor
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


#### Args:

* **tensor** :  The input tensor.

#### Returns:
    Channels, height and width of the `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.