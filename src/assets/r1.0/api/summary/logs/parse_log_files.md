

### parse_log_files
```python
parse_log_files(file_paths:List[str], log_extension:Union[str, NoneType]='.txt', smooth_factor:float=0, save:bool=False, save_path:Union[str, NoneType]=None, ignore_metrics:Union[Set[str], NoneType]=None, share_legend:bool=True, pretty_names:bool=False) -> None
```
Parse one or more log files for graphing.This function which will iterate through the given log file paths, parse them to extract metrics, remove anymetrics which are blacklisted, and then pass the necessary information on the graphing function.

#### Args:

* **file_paths** :  A list of paths to various log files.
* **log_extension** :  The extension of the log files.
* **smooth_factor** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
* **save** :  Whether to save (True) or display (False) the generated graph.
* **save_path** :  Where to save the image if save is true. Defaults to dir_path if not provided.
* **ignore_metrics** :  Any metrics within the log files which will not be visualized.
* **share_legend** :  Whether to have one legend across all graphs (True) or one legend per graph (False).
* **pretty_names** :  Whether to modify the metric names in graph titles (True) or leave them alone (False).

#### Raises:

* **AssertionError** :  If no log files are provided.