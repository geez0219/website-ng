## TerminateOnNaN
```python
TerminateOnNaN(monitor_names=None)
```
End Training if a NaN value is detected. By default (inputs=None) it will monitor all loss values at the endof each batch. If one or more inputs are specified, it will only monitor those values. Inputs may be loss keysand/or the keys corresponding to the outputs of other traces (ex. accuracy) but then the other traces must begiven before TerminateOnNaN in the trace list.

#### Args:

* **monitor_names (str, list, optional)** :  What key(s) to monitor for NaN values.                                        - None (default) will monitor all loss values.                                        - "*" will monitor all state keys and losses.                                        Defaults to None.