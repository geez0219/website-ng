

### parse_log_dir
```python
parse_log_dir(dir_path:str, log_extension:str='.txt', recursive_search:bool=False, smooth_factor:float=1, save:bool=False, save_path:Union[str, NoneType]=None, ignore_metrics:Union[Set[str], NoneType]=None, share_legend:bool=True, pretty_names:bool=False) -> None
```
A function which will gather all log files within a given folder and pass them along for visualization.

#### Args:

* **dir_path** :  The path to a directory containing log files.
* **log_extension** :  The extension of the log files.
* **recursive_search** :  Whether to recursively search sub-directories for log files.
* **smooth_factor** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
* **save** :  Whether to save (True) or display (False) the generated graph.
* **save_path** :  Where to save the image if save is true. Defaults to dir_path if not provided.
* **ignore_metrics** :  Any metrics within the log files which will not be visualized.
* **share_legend** :  Whether to have one legend across all graphs (True) or one legend per graph (False).
* **pretty_names** :  Whether to modify the metric names in graph titles (True) or leave them alone (False).