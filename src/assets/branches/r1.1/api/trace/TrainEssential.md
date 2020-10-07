## TrainEssential<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L134-L181>View source on Github</a>
```python
TrainEssential(
	monitor_names: Set[str]
)
-> None
```
A trace to collect important information during training.

Please don't add this trace into an estimator manually. FastEstimator will add it automatically.


<h3>Args:</h3>

* **monitor_names** :  Which keys from the data dictionary to monitor during training.



