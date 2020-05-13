## TrainEssential
```python
TrainEssential(monitor_names:Set[str]) -> None
```
A trace to collect important information during training.

Please don't add this trace into an estimator manually. FastEstimator will add it automatically.


#### Args:

* **monitor_names** :  Which keys from the data dictionary to monitor during training.

### on_batch_begin
```python
on_batch_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs at the beginning of each batch.


#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.