

### get_current_items
```python
get_current_items(items:Iterable[Union[~T, schedule.schedule.Scheduler[~T]]], run_modes:Union[str, Iterable[str], NoneType]=None, epoch:Union[int, NoneType]=None) -> List[~T]
```
Select items which should be executed for given mode and epoch.



#### Args:

* **items** :  A list of possible items or Schedulers of items to choose from.
* **run_modes** :  The desired execution mode. One or more of "train", "eval", "test", or "infer". If None, items of        all modes will be returned.
* **epoch** :  The desired execution epoch. If None, items across all epochs will be returned.

#### Returns:
    The items which should be executed.