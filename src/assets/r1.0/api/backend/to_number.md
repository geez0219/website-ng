

### to_number
```python
to_number(data:Union[tensorflow.python.framework.ops.Tensor, torch.Tensor, numpy.ndarray, int, float]) -> numpy.ndarray
```
Convert an input value into a Numpy ndarray.
* **This method can be used with Python and Numpy data** : ```pythonb = fe.backend.to_number(5)  # 5 (type==np.ndarray)b = fe.backend.to_number(4.0)  # 4.0 (type==np.ndarray)n = np.array([1, 2, 3])b = fe.backend.to_number(n)  # [1, 2, 3] (type==np.ndarray)```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([1, 2, 3])b = fe.backend.to_number(t)  # [1, 2, 3] (type==np.ndarray)```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([1, 2, 3])b = fe.backend.to_number(p)  # [1, 2, 3] (type==np.ndarray)```

#### Args:

* **data** :  The value to be converted into a np.ndarray.

#### Returns:
    An ndarray corresponding to the given `data`.