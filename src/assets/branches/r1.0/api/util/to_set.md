

### to_set
```python
to_set(data:Any) -> Set[Any]
```
Convert data to a set. A single None value will be converted to the empty set.```pythonx = fe.util.to_set(None)  # {}x = fe.util.to_set([None])  # {None}x = fe.util.to_set(7)  # {7}x = fe.util.to_set([7, 8])  # {7,8}x = fe.util.to_set({7})  # {7}x = fe.util.to_set((7))  # {7}```

#### Args:

* **data** :  Input data, within or without a python container. The `data` must be hashable.

#### Returns:
    The input `data` but inside a set instead of whatever other container type used to hold it.