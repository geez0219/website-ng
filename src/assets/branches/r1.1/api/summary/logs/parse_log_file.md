

### parse_log_file
```python
parse_log_file(
	file_path: str,
	file_extension: str
)
-> fastestimator.summary.summary.Summary
```
A function which will parse log files into a dictionary of metrics.


#### Args:

* **file_path** :  The path to a log file.
* **file_extension** :  The extension of the log file.

#### Returns:
    An experiment summarizing the given log file.