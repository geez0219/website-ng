## TrainEssential
```python
TrainEssential(*args, **kwargs)
```
A trace to collect important information during training.

Please don't add this trace into an estimator manually. FastEstimator will add it automatically.


#### Args:

* **monitor_names** :  Which keys from the data dictionary to monitor during training.