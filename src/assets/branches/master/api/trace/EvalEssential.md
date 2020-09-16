## EvalEssential
```python
EvalEssential(*args, **kwargs)
```
A trace to collect important information during evaluation.

Please don't add this trace into an estimator manually. FastEstimator will add it automatically.


#### Args:

* **monitor_names** :  Any keys which should be collected over the course of an eval epoch.