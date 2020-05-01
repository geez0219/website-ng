

### reduce_mean
```python
reduce_mean(tensor:~Tensor, axis:Union[NoneType, int, Sequence[int]]=None, keepdims:bool=False) -> ~Tensor
```
Compute the mean value along a given `axis` of a `tensor`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_mean(n)  # 4.5b = fe.backend.reduce_mean(n, axis=0)  # [[3, 4], [5, 6]]b = fe.backend.reduce_mean(n, axis=1)  # [[2, 3], [6, 7]]b = fe.backend.reduce_mean(n, axis=[0,2])  # [3.5, 5.5]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_mean(t)  # 4.5b = fe.backend.reduce_mean(t, axis=0)  # [[3, 4], [5, 6]]b = fe.backend.reduce_mean(t, axis=1)  # [[2, 3], [3, 7]]b = fe.backend.reduce_mean(t, axis=[0,2])  # [3.5, 5.5]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_mean(p)  # 4.5b = fe.backend.reduce_mean(p, axis=0)  # [[3, 4], [5, 6]]b = fe.backend.reduce_mean(p, axis=1)  # [[2, 3], [6, 7]]b = fe.backend.reduce_mean(p, axis=[0,2])  # [3.5, 5.5]```

#### Args:

* **tensor** :  The input value.
* **axis** :  Which axis or collection of axes to compute the mean along.
* **keepdims** :  Whether to preserve the number of dimensions during the reduction.

#### Returns:
    The mean values of `tensor` along `axis`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.