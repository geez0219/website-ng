

### clip_by_value
```python
clip_by_value(tensor:~Tensor, min_value:Union[int, float, ~Tensor], max_value:Union[int, float, ~Tensor]) -> ~Tensor
```
Clip a tensor such that `min_value` <= tensor <= `max_value`.
* **This method can be used with Numpy data** : ```pythonn = np.array([-5, 4, 2, 0, 9, -2])b = fe.backend.clip_by_value(n, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([-5, 4, 2, 0, 9, -2])b = fe.backend.clip_by_value(t, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([-5, 4, 2, 0, 9, -2])b = fe.backend.clip_by_value(p, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]```

#### Args:

* **tensor** :  The input value.
* **min_value** :  The minimum value to clip to.
* **max_value** :  The maximum value to clip to.

#### Returns:
    The `tensor`, with it's values clipped.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.