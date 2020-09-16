

### get_signature_epochs
```python
get_signature_epochs(items:List[Any], total_epochs:int, mode:Union[str, NoneType]=None) -> List[int]
```
Find all epochs of changes due to schedulers.


#### Args:

* **items** :  List of items to scan from.
* **total_epochs** :  The maximum epoch number to consider when searching for signature epochs.
* **mode** :  Current execution mode. If None, all execution modes will be considered.

#### Returns:
    The epoch numbers of changes.