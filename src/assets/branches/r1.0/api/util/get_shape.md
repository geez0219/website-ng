

### get_shape
```python
get_shape(obj:Any) -> List[Union[int, NoneType]]
```
A function to find the shapes of an object or sequence of objects.Lists or Tuples will assume that the zeroth dimension is ragged (shape==None). If entries in the list havemismatched ranks, then only the list dimension will be considered as part of the shape. If all ranks are equal, anattempt will be made to determine which of the interior dimensions are ragged.```pythonx = fe.util.get_shape(np.ones((12,22,11)))  # [12, 22, 11]x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5))])  # [None]x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5, 4))])  # [None, None, None, None]x = fe.util.get_shape([np.ones((12,22,11)), np.ones((12, 22, 4))])  # [None, 12, 22, None]
* **x = fe.util.get_shape({"a"** :  np.ones((12,22,11))})  # []```

#### Args:

* **obj** :  Data to infer the shape of.

#### Returns:
    A list representing the shape of the data.