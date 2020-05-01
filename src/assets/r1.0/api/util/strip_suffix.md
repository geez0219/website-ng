

### strip_suffix
```python
strip_suffix(target:Union[str, NoneType], suffix:Union[str, NoneType]) -> Union[str, NoneType]
```
Remove the given `suffix` from the `target` if it is present there.```pythonx = fe.util.strip_suffix("astring.json", ".json")  # "astring"x = fe.util.strip_suffix("astring.json", ".yson")  # "astring.json"```

#### Args:

* **target** :  A string to be formatted.
* **suffix** :  A string to be removed from `target`.

#### Returns:
    The formatted version of `target`.