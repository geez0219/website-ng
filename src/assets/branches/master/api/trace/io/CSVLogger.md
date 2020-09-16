## CSVLogger
```python
CSVLogger(*args, **kwargs)
```
Log monitored quantities in a CSV file.


#### Args:

* **filename** :  Output filename.
* **monitor_names** :  List of keys to monitor. If None then all metrics will be recorded.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".