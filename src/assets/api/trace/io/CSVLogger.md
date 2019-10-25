## CSVLogger
```python
CSVLogger(filename, monitor_names=None, separator=', ', append=False, mode='eval')
```
Log monitored quantity in CSV file manner

#### Args:

* **filename (str)** :  Output filename.
* **monitor_names (list of str, optional)** :  List of key names to monitor. The names can be {"mode", "epoch",        "train_step", or output names that other traces create}. If None, it will record all. Defaults to None.
* **separator (str, optional)** :  Seperator for numbers. Defaults to ", ".
* **append (bool, optional)** :  If true, it will write csv file in append mode. Otherwise, it will overwrite the        existed file. Defaults to False.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always        execute. Defaults to 'eval'.