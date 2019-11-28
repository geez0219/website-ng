

### parse_log_files
```python
parse_log_files(file_paths, log_extension='.txt', smooth_factor=0, save=False, save_path=None, ignore_metrics=None, share_legend=True, pretty_names=False)
```
A function which will iterate through the given log file paths, parse them to extract metrics, remove anymetrics which are blacklisted, and then pass the necessary information on the graphing function

#### Args:

* **file_paths** :  A list of paths to various log files
* **log_extension** :  The extension of the log files
* **smooth_factor** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none)
* **save** :  Whether to save (true) or display (false) the generated graph
* **save_path** :  Where to save the image if save is true. Defaults to dir_path if not provided
* **ignore_metrics** :  Any metrics within the log files which will not be visualized
* **share_legend** :  Whether to have one legend across all graphs (true) or one legend per graph (false)
* **pretty_names** :  Whether to modify the metric names in graph titles (true) or leave them alone (false)

#### Returns:
    None