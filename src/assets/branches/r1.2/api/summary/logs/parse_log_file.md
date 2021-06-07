## parse_log_file<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/summary/logs/log_parse.py/#L26-L80>View source on Github</a>
```python
parse_log_file(
	file_path: str,
	file_extension: str
)
-> fastestimator.summary.summary.Summary
```
A function which will parse log files into a dictionary of metrics.


<h3>Args:</h3>


* **file_path**: The path to a log file.

* **file_extension**: The extension of the log file.


<h3>Returns:</h3>

<ul class="return-block"><li>    An experiment summarizing the given log file.</li></ul>

