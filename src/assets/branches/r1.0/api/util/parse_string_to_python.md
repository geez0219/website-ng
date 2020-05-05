

### parse_string_to_python
```python
parse_string_to_python(val:str) -> Any
```
Convert a string into a python object.```pythonx = fe.util.parse_string_to_python("5")  # 5x = fe.util.parse_string_to_python("[5, 4, 0.3]")  # [5, 4, 0.3]
* **x = fe.util.parse_string_to_python("{'a'** : 5, 'b'7}")  # {'a'5, 'b'7}```

#### Args:

* **val** :  An input string.

#### Returns:
    A python object version of the input string.