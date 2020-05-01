

### to_list
```python
to_list(data:Any) -> List[Any]
```
Convert data to a list. A single None value will be converted to the empty list.```pythonx = fe.util.to_list(None)  # []x = fe.util.to_list([None])  # [None]x = fe.util.to_list(7)  # [7]x = fe.util.to_list([7, 8])  # [7,8]x = fe.util.to_list({7})  # [7]x = fe.util.to_list((7))  # [7]
* **x = fe.util.to_list({'a'** :  7})  # [{'a' 7}]```

#### Args:

* **data** :  Input data, within or without a python container.

#### Returns:
    The input `data` but inside a list instead of whatever other container type used to hold it.