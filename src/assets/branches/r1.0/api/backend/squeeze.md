

### squeeze
```python
squeeze(tensor:~Tensor, axis:Union[int, NoneType]=None) -> ~Tensor
```
Remove an `axis` from a `tensor` if that axis has length 1.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)b = fe.backend.squeeze(n)  # [[1, 2], [3, 4], [5, 6]]b = fe.backend.squeeze(n, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]b = fe.backend.squeeze(n, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)b = fe.backend.squeeze(t)  # [[1, 2], [3, 4], [5, 6]]b = fe.backend.squeeze(t, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]b = fe.backend.squeeze(t, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)b = fe.backend.squeeze(p)  # [[1, 2], [3, 4], [5, 6]]b = fe.backend.squeeze(p, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]b = fe.backend.squeeze(p, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]```

#### Args:

* **tensor** :  The input value.
* **axis** :  Which axis to squeeze along, which must have length==1 (or pass None to squeeze all length 1 axes).

#### Returns:
    The reshaped `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.