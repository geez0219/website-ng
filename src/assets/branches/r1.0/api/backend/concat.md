

### concat
```python
concat(tensors:List[~Tensor], axis:int=0) -> Union[~Tensor, NoneType]
```
Concatenate a list of `tensors` along a given `axis`.
* **This method can be used with Numpy data** : ```pythonn = [np.array([[0, 1]]), np.array([[2, 3]]), np.array([[4, 5]])]b = fe.backend.concat(n, axis=0)  # [[0, 1], [2, 3], [4, 5]]b = fe.backend.concat(n, axis=1)  # [[0, 1, 2, 3, 4, 5]]```
* **This method can be used with TensorFlow tensors** : ```pythont = [tf.constant([[0, 1]]), tf.constant([[2, 3]]), tf.constant([[4, 5]])]b = fe.backend.concat(t, axis=0)  # [[0, 1], [2, 3], [4, 5]]b = fe.backend.concat(t, axis=1)  # [[0, 1, 2, 3, 4, 5]]```
* **This method can be used with PyTorch tensors** : ```pythonp = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])]b = fe.backend.concat(p, axis=0)  # [[0, 1], [2, 3], [4, 5]]b = fe.backend.concat(p, axis=1)  # [[0, 1, 2, 3, 4, 5]]```

#### Args:

* **tensors** :  A list of tensors to be concatenated.
* **axis** :  The axis along which to concatenate the input.

#### Returns:
    A concatenated representation of the `tensors`, or None if the list of `tensors` was empty.

#### Raises:

* **ValueError** :  If `tensors` is an unacceptable data type.