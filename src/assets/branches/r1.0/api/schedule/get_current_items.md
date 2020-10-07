## get_current_items<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L165-L193>View source on Github</a>
```python
get_current_items(
	items: Iterable[Union[~T, fastestimator.schedule.schedule.Scheduler[~T]]],
	run_modes: Union[str, Iterable[str], NoneType]=None,
	epoch: Union[int, NoneType]=None
)
-> List[~T]
```
Select items which should be executed for given mode and epoch.


<h3>Args:</h3>

* **items** :  A list of possible items or Schedulers of items to choose from.
* **run_modes** :  The desired execution mode. One or more of "train", "eval", "test", or "infer". If None, items of        all modes will be returned.
* **epoch** :  The desired execution epoch. If None, items across all epochs will be returned.

<h3>Returns:</h3>
    The items which should be executed.

