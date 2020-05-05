

### pad_batch
```python
pad_batch(batch:List[MutableMapping[str, Any]], pad_value:Union[float, int]) -> None
```
A function to pad a batch of data in-place by appending to the ends of the tensors.```python
* **data = [{"x"** :  np.ones((2, 2)), "y" 8}, {"x" np.ones((3, 1)), "y" 4}]fe.util.pad_batch(data, pad_value=0)
* **print(data)  # [{'x'** :  [[1., 1.], [1., 1.],[0., 0.]], 'y' 8}, {'x' [[1., 0.], [1., 0.], [1., 0.]]), 'y' 4}]```

#### Args:

* **batch** :  A list of data to be padded.
* **pad_value** :  The value to pad with.

#### Raises:

* **AssertionError** :  If the data within the batch do not have matching ranks.