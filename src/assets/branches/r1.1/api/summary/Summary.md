## Summary<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/summary/summary.py/#L21-L78>View source on Github</a>
```python
Summary(
	name: Union[str, NoneType],
	system_config: Union[List[fastestimator.util.traceability_util.FeSummaryTable], NoneType]=None
)
-> None
```
A summary object that records training history.

This class is intentionally not @traceable.


<h3>Args:</h3>


* **name**: Name of the experiment. If None then experiment results will be ignored.

* **system_config**: A description of the initialization parameters defining the estimator associated with this experiment.

---

### merge<span class="tag">method of Summary</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/summary/summary.py/#L36-L44>View source on Github</a>
```python
merge(
	self,
	other: 'Summary'
)
```
Merge another `Summary` into this one.


<h4>Args:</h4>


* **other**: Other `summary` object to be merged.

