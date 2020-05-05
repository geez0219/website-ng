

### permute
```python
permute(tensor:~Tensor, permutation:List[int]) -> ~Tensor
```
Perform the specified `permutation` on the axes of a given `tensor`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])b = fe.backend.permute(n, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]b = fe.backend.permute(n, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])b = fe.backend.permute(t, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]b = fe.backend.permute(t, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])b = fe.backend.permute(p, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]b = fe.backend.permute(P, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]```

#### Args:

* **tensor** :  The tensor to permute.
* **permutation** :  The new axis order to be used. Should be a list containing all integers in range [0, tensor.ndim).

#### Returns:
    The `tensor` with axes swapped according to the `permutation`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.