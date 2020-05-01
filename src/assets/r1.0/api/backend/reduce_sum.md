

### reduce_sum
```python
reduce_sum(tensor:~Tensor, axis:Union[NoneType, int, Sequence[int]]=None, keepdims:bool=False) -> ~Tensor
```
Compute the sum along a given `axis` of a `tensor`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_sum(n)  # 36b = fe.backend.reduce_sum(n, axis=0)  # [[6, 8], [10, 12]]b = fe.backend.reduce_sum(n, axis=1)  # [[4, 6], [12, 14]]b = fe.backend.reduce_sum(n, axis=[0,2])  # [14, 22]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_sum(t)  # 36b = fe.backend.reduce_sum(t, axis=0)  # [[6, 8], [10, 12]]b = fe.backend.reduce_sum(t, axis=1)  # [[4, 6], [12, 14]]b = fe.backend.reduce_sum(t, axis=[0,2])  # [14, 22]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reduce_sum(p)  # 36b = fe.backend.reduce_sum(p, axis=0)  # [[6, 8], [10, 12]]b = fe.backend.reduce_sum(p, axis=1)  # [[4, 6], [12, 14]]b = fe.backend.reduce_sum(p, axis=[0,2])  # [14, 22]```

#### Args:

* **tensor** :  The input value.
* **axis** :  Which axis or collection of axes to compute the sum along.
* **keepdims** :  Whether to preserve the number of dimensions during the reduction.

#### Returns:
    The sum of `tensor` along `axis`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.