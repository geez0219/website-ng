

### abs
```python
abs(tensor:~Tensor) -> ~Tensor
```
Compute the absolute value of a tensor.
* **This method can be used with Numpy data** : ```pythonn = np.array([-2, 7, -19])b = fe.backend.abs(n)  # [2, 7, 19]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([-2, 7, -19])b = fe.backend.abs(t)  # [2, 7, 19]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([-2, 7, -19])b = fe.backend.abs(p)  # [2, 7, 19]```

#### Args:

* **tensor** :  The input value.

#### Returns:
    The absolute value of `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.