## TerminateOnNaN
```python
TerminateOnNaN(
	monitor_names: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Set[str]]=None
)
-> None
```
End Training if a NaN value is detected.

By default (monitor_names=None) it will monitor all loss values at the end of each batch. If one or more inputs are
specified, it will only monitor those values. Inputs may be loss keys and/or the keys corresponding to the outputs
of other traces (ex. accuracy).


#### Args:

* **monitor_names** :  key(s) to monitor for NaN values. If None, all loss values will be monitored. "*" will monitor        all trace output keys and losses.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".