

### percentile
```python
percentile(tensor:~Tensor, percentiles:Union[int, List[int]], axis:Union[NoneType, int, List[int]]=None, keepdims:bool=True) -> ~Tensor
```
Compute the `percentiles` of a `tensor`.The n-th percentile of `tensor` is the value n/100 of the way from the minimum to the maximum in a sorted copy of`tensor`. If the percentile falls in between two values, the nearest of the two values will be used.
* **This method can be used with Numpy data** : ```pythonn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])b = fe.backend.percentile(n, percentiles=[66])  # [[[6]]]b = fe.backend.percentile(n, percentiles=[66], axis=0)  # [[[4, 5, 6]]]b = fe.backend.percentile(n, percentiles=[66], axis=1)  # [[[2], [5], [8]]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])b = fe.backend.percentile(t, percentiles=[66])  # [[[6]]]b = fe.backend.percentile(t, percentiles=[66], axis=0)  # [[[4, 5, 6]]]b = fe.backend.percentile(t, percentiles=[66], axis=1)  # [[[2], [5], [8]]]```
* **This method can be used with PyTorch tensors** : ```pythonp = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])b = fe.backend.percentile(p, percentiles=[66])  # [[[6]]]b = fe.backend.percentile(p, percentiles=[66], axis=0)  # [[[4, 5, 6]]]b = fe.backend.percentile(p, percentiles=[66], axis=1)  # [[[2], [5], [8]]]```

#### Args:

* **tensor** :  The tensor from which to extract percentiles.
* **percentiles** :  One or more percentile values to be computed.
* **axis** :  Along which axes to compute the percentile (None to compute over all axes).
* **keepdims** :  Whether to maintain the number of dimensions from `tensor`.

#### Returns:
    The `percentiles` of the given `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.