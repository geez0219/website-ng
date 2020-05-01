

### to_shape
```python
to_shape(data:Union[Collection, ~Tensor], add_batch=False, exact_shape=True) -> Union[Collection, ~Tensor]
```
Compute the shape of tensors within a collection of `data`.
* **This method can be used with Numpy data** : ```python
* **data = {"x"** :  np.ones((10,15)), "y"[np.ones((4)), np.ones((5, 3))], "z"{"key"np.ones((2,2))}}
* **shape = fe.backend.to_shape(data)  # {"x"** :  (10, 15), "y"[(4), (5, 3)], "z" {"key" (2, 2)}}shape = fe.backend.to_shape(data, add_batch=True)
* **# {"x"** :  (None, 10, 15), "y"[(None, 4), (None, 5, 3)], "z" {"key" (None, 2, 2)}}shape = fe.backend.to_shape(data, exact_shape=False)
* **# {"x"** :  (None, None), "y"[(None), (None, None)], "z" {"key" (None, None)}}```
* **This method can be used with TensorFlow tensors** : ```python
* **data = {"x"** :  tf.ones((10,15)), "y"[tf.ones((4)), tf.ones((5, 3))], "z"{"key"tf.ones((2,2))}}
* **shape = fe.backend.to_shape(data)  # {"x"** :  (10, 15), "y"[(4), (5, 3)], "z" {"key" (2, 2)}}shape = fe.backend.to_shape(data, add_batch=True)
* **# {"x"** :  (None, 10, 15), "y"[(None, 4), (None, 5, 3)], "z" {"key" (None, 2, 2)}}shape = fe.backend.to_shape(data, exact_shape=False)
* **# {"x"** :  (None, None), "y"[(None), (None, None)], "z" {"key" (None, None)}}```
* **This method can be used with PyTorch tensors** : ```python
* **data = {"x"** :  torch.ones((10,15)), "y"[torch.ones((4)), torch.ones((5, 3))], "z"{"key"torch.ones((2,2))}}
* **shape = fe.backend.to_shape(data)  # {"x"** :  (10, 15), "y"[(4), (5, 3)], "z" {"key" (2, 2)}}shape = fe.backend.to_shape(data, add_batch=True)
* **# {"x"** :  (None, 10, 15), "y"[(None, 4), (None, 5, 3)], "z" {"key" (None, 2, 2)}}shape = fe.backend.to_shape(data, exact_shape=False)
* **# {"x"** :  (None, None), "y"[(None), (None, None)], "z" {"key" (None, None)}}```

#### Args:

* **data** :  A tensor or possibly nested collection of tensors.
* **add_batch** :  Whether to prepend a batch dimension to the shapes.
* **exact_shape** :  Whether to return the exact shapes, or if False to fill the shapes with None values.

#### Returns:
    A collection with the same structure as `data`, but with any tensors substituted for their shapes.