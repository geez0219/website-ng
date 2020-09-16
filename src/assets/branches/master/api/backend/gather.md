

### gather
```python
gather(tensor:~Tensor, indices:~Tensor) -> ~Tensor
```
Gather specific indices from a tensor.

The `indices` will automatically be cast to the correct type (tf, torch, np) based on the type of the `tensor`.

This method can be used with Numpy data:
```python
ind = np.array([1, 0, 1])
n = np.array([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather(n, ind)  # [[2, 3], [0, 1], [2, 3]]
n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather(n, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
```

This method can be used with TensorFlow tensors:
```python
ind = tf.constant([1, 0, 1])
t = tf.constant([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather(t, ind)  # [[2, 3], [0, 1], [2, 3]]
t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather(t, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
```

This method can be used with PyTorch tensors:
```python
ind = torch.tensor([1, 0, 1])
p = torch.tensor([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather(p, ind)  # [[2, 3], [0, 1], [2, 3]]
p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather(p, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
```


#### Args:

* **tensor** :  A tensor to gather values from.
* **indices** :  A tensor indicating which indices should be selected. These represent locations along the 0 axis.

#### Returns:
    A tensor containing the elements from `tensor` at the given `indices`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.