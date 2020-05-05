## DefaultKeyDict
```python
DefaultKeyDict(default:Callable[[Any], Any], **kwargs) -> None
```
Like collections.defaultdict but it passes the key argument to the default function.    ```python
* **d = fe.util.DefaultKeyDict(default=lambda x** :  x+x, a=4, b=6)    print(d["a"])  # 4    print(d["c"])  # "cc"    ```

#### Args:

* **default** :  A function which takes a key and returns a default value based on the key.
 **kwargs :  Initial key/value pairs for the dictionary.    