

### to_type
```python
to_type(data:Union[Collection, ~Tensor]) -> Collection
```
Compute the data types of tensors within a collection of `data`.
* **This method can be used with Numpy data** : ```python
* **data = {"x"** :  np.ones((10,15), dtype="float32"), "y"[np.ones((4), dtype="int8"), np.ones((5, 3), dtype="double")],
* **"z"** : {"key"np.ones((2,2), dtype="int64")}}types = fe.backend.to_type(data)
* **# {'x'** :  dtype('float32'), 'y' [dtype('int8'), dtype('float64')], 'z' {'key' dtype('int64')}}```
* **This method can be used with TensorFlow tensors** : ```python
* **data = {"x"** :  tf.ones((10,15), dtype="float32"), "y"[tf.ones((4), dtype="int8"), tf.ones((5, 3), dtype="double")],
* **"z"** : {"key"tf.ones((2,2), dtype="int64")}}types = fe.backend.to_type(data)
* **# {'x'** :  tf.float32, 'y' [tf.int8, tf.float64], 'z' {'key' tf.int64}}```
* **This method can be used with PyTorch tensors** : ```python
* **data = {"x"** :  torch.ones((10,15), dtype=torch.float32), "y"[torch.ones((4), dtype=torch.int8), torch.ones((5, 3),
* **dtype=torch.double)], "z"** : {"key"torch.ones((2,2), dtype=torch.long)}}types = fe.backend.to_type(data)
* **# {'x'** :  torch.float32, 'y' [torch.int8, torch.float64], 'z' {'key' torch.int64}}```

#### Args:

* **data** :  A tensor or possibly nested collection of tensors.

#### Returns:
    A collection with the same structure as `data`, but with any tensors substituted for their dtypes.