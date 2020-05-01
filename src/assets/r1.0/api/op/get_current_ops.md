

### get_current_ops
```python
get_current_ops(ops:Iterable[Union[~OpType, fastestimator.schedule.schedule.Scheduler[~OpType]]], mode:str, epoch:int=0) -> List[~OpType]
```
Select ops which should be executed for given mode and epoch.

#### Args:

* **ops** :  A list of possible Ops or Schedulers of Ops to choose from.
* **mode** :  The desired execution mode. One of "train", "eval", "test", or "infer".
* **epoch** :  The desired execution epoch.

#### Returns:
    The `Ops` which should be executed.