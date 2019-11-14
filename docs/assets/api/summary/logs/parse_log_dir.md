

### parse_log_dir
```python
parse_log_dir(dir_path, log_extension='.txt', recursive_search=False, smooth_factor=1, save=False, save_path=None, ignore_metrics=None, share_legend=True, pretty_names=False)
```
A function which will gather all log files within a given folder and pass them along for visualization

#### Args:

* **dir_path** :  The path to a directory containing log files
* **log_extension** :  The extension of the log files
* **recursive_search** :  Whether to recursively search sub-directories for log files
* **smooth_factor** :  A non-negative float representing the magnitude of gaussian smoothing to apply(zero for none)
* **save** :  Whether to save (true) or display (false) the generated graph
* **save_path** :  Where to save the image if save is true. Defaults to dir_path if not provided
* **ignore_metrics** :  Any metrics within the log files which will not be visualized
* **share_legend** :  Whether to have one legend across all graphs (true) or one legend per graph (false)
* **pretty_names** :  Whether to modify the metric names in graph titles (true) or leave them alone (false)

#### Returns:
    None