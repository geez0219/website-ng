

### reshape
```python
reshape(tensor:~Tensor, shape:List[int]) -> ~Tensor
```
Reshape a `tensor` to conform to a given shape.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reshape(n, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]b = fe.backend.reshape(n, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]b = fe.backend.reshape(n, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]b = fe.backend.reshape(n, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reshape(t, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]b = fe.backend.reshape(t, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]b = fe.backend.reshape(t, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]b = fe.backend.reshape(t, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])b = fe.backend.reshape(p, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]b = fe.backend.reshape(p, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]b = fe.backend.reshape(p, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]b = fe.backend.reshape(p, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]```

#### Args:

* **tensor** :  The input value.
* **shape** :  The new shape of the tensor. At most one value may be -1 which indicates that whatever values are left        should be packed into that axis.

#### Returns:
    The reshaped `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.