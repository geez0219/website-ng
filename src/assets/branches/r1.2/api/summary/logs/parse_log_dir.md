## parse_log_dir<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/summary/logs/log_parse.py/#L139-L178>View source on Github</a>
```python
parse_log_dir(
	dir_path: str,
	log_extension: str='.txt',
	recursive_search: bool=False,
	smooth_factor: float=1,
	save: bool=False,
	save_path: Union[str, NoneType]=None,
	ignore_metrics: Union[Set[str], NoneType]=None,
	include_metrics: Union[Set[str], NoneType]=None,
	share_legend: bool=True,
	pretty_names: bool=False,
	group_by: Union[str, NoneType]=None
)
-> None
```
A function which will gather all log files within a given folder and pass them along for visualization.


<h3>Args:</h3>


* **dir_path**: The path to a directory containing log files.

* **log_extension**: The extension of the log files.

* **recursive_search**: Whether to recursively search sub-directories for log files.

* **smooth_factor**: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).

* **save**: Whether to save (True) or display (False) the generated graph.

* **save_path**: Where to save the image if save is true. Defaults to dir_path if not provided.

* **ignore_metrics**: Any metrics within the log files which will not be visualized.

* **include_metrics**: A whitelist of metric keys (None whitelists all keys).

* **share_legend**: Whether to have one legend across all graphs (True) or one legend per graph (False).

* **pretty_names**: Whether to modify the metric names in graph titles (True) or leave them alone (False).

* **group_by**: Combine multiple log files by a regex to visualize their mean+-stddev. For example, to group together files like [a_1.txt, a_2.txt] vs [b_1.txt, b_2.txt] you can use: r'(.*)_[\d]+\.txt'.

