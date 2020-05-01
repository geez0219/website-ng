

### zeros_like
```python
zeros_like(tensor:~Tensor, dtype:Union[NoneType, str]=None) -> ~Tensor
```
Generate zeros shaped like `tensor` with a specified `dtype`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[0,1],[2,3]])b = fe.backend.zeros_like(n)  # [[0, 0], [0, 0]]b = fe.backend.zeros_like(n, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[0,1],[2,3]])b = fe.backend.zeros_like(t)  # [[0, 0], [0, 0]]b = fe.backend.zeros_like(t, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[0,1],[2,3]])b = fe.backend.zeros_like(p)  # [[0, 0], [0, 0]]b = fe.backend.zeros_like(p, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]```

#### Args:

* **tensor** :  The tensor whose shape will be copied.
* **dtype** :  The data type to be used when generating the resulting tensor. If None then the `tensor` dtype is used.

#### Returns:
    A tensor of zeros with the same shape as `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.