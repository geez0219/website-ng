## parse_log_files<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/summary/logs/log_parse.py/#L83-L136>View source on Github</a>
```python
parse_log_files(
	file_paths: List[str],
	log_extension: Union[str, NoneType]='.txt',
	smooth_factor: float=0,
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
Parse one or more log files for graphing.

This function which will iterate through the given log file paths, parse them to extract metrics, remove any
metrics which are blacklisted, and then pass the necessary information on the graphing function.


<h3>Args:</h3>


* **file_paths**: A list of paths to various log files.

* **log_extension**: The extension of the log files.

* **smooth_factor**: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).

* **save**: Whether to save (True) or display (False) the generated graph.

* **save_path**: Where to save the image if save is true. Defaults to dir_path if not provided.

* **ignore_metrics**: Any metrics within the log files which will not be visualized.

* **include_metrics**: A whitelist of metric keys (None whitelists all keys).

* **share_legend**: Whether to have one legend across all graphs (True) or one legend per graph (False).

* **pretty_names**: Whether to modify the metric names in graph titles (True) or leave them alone (False).

* **group_by**: Combine multiple log files by a regex to visualize their mean+-stddev. For example, to group together files like [a_1.txt, a_2.txt] vs [b_1.txt, b_2.txt] you can use: r'(.*)_[\d]+\.txt'. 

<h3>Raises:</h3>


* **AssertionError**: If no log files are provided.

* **ValueError**: If a log file does not match the `group_by` regex pattern.

