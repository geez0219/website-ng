

### reduce_max
```python
reduce_max(tensor:~Tensor, axis:Union[NoneType, int, Sequence[int]]=None, keepdims:bool=False) -> ~Tensor
```
Compute the maximum value along a given `axis` of a `tensor`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])b = fe.backend.reduce_max(n)  # 8b = fe.backend.reduce_max(n, axis=0)  # [[5, 6], [7, 8]]b = fe.backend.reduce_max(n, axis=1)  # [[3, 4], [7, 8]]b = fe.backend.reduce_max(n, axis=[0,2])  # [6, 8]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])b = fe.backend.reduce_max(t)  # 8b = fe.backend.reduce_max(t, axis=0)  # [[5, 6], [7, 8]]b = fe.backend.reduce_max(t, axis=1)  # [[3, 4], [7, 8]]b = fe.backend.reduce_max(t, axis=[0,2])  # [6, 8]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])b = fe.backend.reduce_max(p)  # 8b = fe.backend.reduce_max(p, axis=0)  # [[5, 6], [7, 8]]b = fe.backend.reduce_max(p, axis=1)  # [[3, 4], [7, 8]]b = fe.backend.reduce_max(p, axis=[0,2])  # [6, 8]```

#### Args:

* **tensor** :  The input value.
* **axis** :  Which axis or collection of axes to compute the maximum along.
* **keepdims** :  Whether to preserve the number of dimensions during the reduction.

#### Returns:
    The maximum values of `tensor` along `axis`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.