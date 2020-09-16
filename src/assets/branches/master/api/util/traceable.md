

### traceable
```python
traceable(whitelist:Union[str, Tuple[str]]=(), blacklist:Union[str, Tuple[str]]=()) -> Callable
```
A decorator to be placed on classes in order to make them traceable and to enable a deep restore.

Decorated classes will gain the .fe_summary() and .fe_state() methods.


#### Args:

* **whitelist** :  Arguments which should be included in a deep restore of the decorated class.
* **blacklist** :  Arguments which should be excluded from a deep restore of the decorated class.

#### Returns:
    The decorated class.