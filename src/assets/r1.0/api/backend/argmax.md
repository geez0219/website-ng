

### argmax
```python
argmax(tensor:~Tensor, axis:int=0) -> ~Tensor
```
Compute the index of the maximum value along a given axis of a tensor.
* **This method can be used with Numpy data** : ```pythonn = np.array([[2,7,5],[9,1,3],[4,8,2]])b = fe.backend.argmax(n, axis=0)  # [1, 2, 0]b = fe.backend.argmax(n, axis=1)  # [1, 0, 1]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[2,7,5],[9,1,3],[4,8,2]])b = fe.backend.argmax(t, axis=0)  # [1, 2, 0]b = fe.backend.argmax(t, axis=1)  # [1, 0, 1]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[2,7,5],[9,1,3],[4,8,2]])b = fe.backend.argmax(p, axis=0)  # [1, 2, 0]b = fe.backend.argmax(p, axis=1)  # [1, 0, 1]```

#### Args:

* **tensor** :  The input value.
* **axis** :  Which axis to compute the index along.

#### Returns:
    The indices corresponding to the maximum values within `tensor` along `axis`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.