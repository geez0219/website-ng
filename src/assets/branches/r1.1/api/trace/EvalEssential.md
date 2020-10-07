## EvalEssential<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L185-L210>View source on Github</a>
```python
EvalEssential(
	monitor_names: Set[str]
)
-> None
```
A trace to collect important information during evaluation.

Please don't add this trace into an estimator manually. FastEstimator will add it automatically.


<h3>Args:</h3>

* **monitor_names** :  Any keys which should be collected over the course of an eval epoch.



