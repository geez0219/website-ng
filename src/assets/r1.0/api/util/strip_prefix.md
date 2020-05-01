

### strip_prefix
```python
strip_prefix(target:Union[str, NoneType], prefix:Union[str, NoneType]) -> Union[str, NoneType]
```
Remove the given `prefix` from the `target` if it is present there.```pythonx = fe.util.strip_prefix("astring.json", "ast")  # "ring.json"x = fe.util.strip_prefix("astring.json", "asa")  # "astring.json"```

#### Args:

* **target** :  A string to be formatted.
* **prefix** :  A string to be removed from `target`.

#### Returns:
    The formatted version of `target`.