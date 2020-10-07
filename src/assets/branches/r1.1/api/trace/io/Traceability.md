## Traceability<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/traceability.py/#L67-L673>View source on Github</a>
```python
Traceability(
	save_path: str,
	extra_objects: Any=None
)
```
Automatically generate summary reports of the training.


<h3>Args:</h3>

* **save_path** :  Where to save the output files. Note that this will generate a new folder with the given name, into        which the report and corresponding graphics assets will be written.
* **extra_objects** :  Any extra objects which are not part of the Estimator, but which you want to capture in the        summary report. One example could be an extra pipeline which performs pre-processing.

<h3>Raises:</h3>

* **OSError** :  If graphviz is not installed.



