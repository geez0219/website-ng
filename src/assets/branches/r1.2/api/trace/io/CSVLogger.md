## CSVLogger<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/io/csv_logger.py/#L27-L63>View source on Github</a>
```python
CSVLogger(
	filename: str,
	monitor_names: Union[List[str], str, NoneType]=None,
	mode: Union[str, Set[str]]=('eval', 'test')
)
-> None
```
Log monitored quantities in a CSV file.


<h3>Args:</h3>


* **filename**: Output filename.

* **monitor_names**: List of keys to monitor. If None then all metrics will be recorded.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

