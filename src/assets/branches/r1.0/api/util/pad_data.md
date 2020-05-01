

### pad_data
```python
pad_data(data:numpy.ndarray, target_shape:Tuple[int, ...], pad_value:Union[float, int]) -> numpy.ndarray
```
Pad `data` by appending `pad_value`s along it's dimensions until the `target_shape` is reached.```pythonx = np.ones((1,2))x = fe.util.pad_data(x, target_shape=(3, 3), pad_value = -2)  # [[1, 1, -2], [-2, -2, -2], [-2, -2, -2]]```

#### Args:

* **data** :  The data to be padded.
* **target_shape** :  The desired shape for `data`. Should have the same rank as `data`, with each dimension being >=        the size of the `data` dimension.
* **pad_value** :  The value to insert into `data` if padding is required to achieve the `target_shape`.

#### Returns:
    The `data`, padded to the `target_shape`.