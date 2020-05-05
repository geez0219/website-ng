

### expand_dims
```python
expand_dims(tensor:~Tensor, axis:int=1) -> ~Tensor
```
Create a new dimension in `tensor` along a given `axis`.
* **This method can be used with Numpy data** : ```pythonn = np.array([2,7,5])b = fe.backend.expand_dims(n, axis=0)  # [[2, 5, 7]]b = fe.backend.expand_dims(n, axis=1)  # [[2], [5], [7]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([2,7,5])b = fe.backend.expand_dims(t, axis=0)  # [[2, 5, 7]]b = fe.backend.expand_dims(t, axis=1)  # [[2], [5], [7]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([2,7,5])b = fe.backend.expand_dims(p, axis=0)  # [[2, 5, 7]]b = fe.backend.expand_dims(p, axis=1)  # [[2], [5], [7]]```

#### Args:

* **tensor** :  The input to be modified, having n dimensions.
* **axis** :  Which axis should the new axis be inserted along. Must be in the range [-n-1, n].

#### Returns:
    A concatenated representation of the `tensors`, or None if the list of `tensors` was empty.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.