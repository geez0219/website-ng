

### check_nan
```python
check_nan(val:Union[int, float, numpy.ndarray, tensorflow.python.framework.ops.Tensor, torch.Tensor]) -> bool
```
Checks if the input contains NaN values.

This method can be used with Numpy data:
```python
n = np.array([[[1.0, 2.0], [3.0, np.NaN]], [[5.0, 6.0], [7.0, 8.0]]])
b = fe.backend.check_nan(n)  # True
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[np.NaN, 6.0], [7.0, 8.0]]])
b = fe.backend.check_nan(n)  # True
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [np.NaN, 8.0]]])
b = fe.backend.check_nan(n)  # True
```


#### Args:

* **val** :  The input value.

#### Returns:
    True iff `val` contains NaN