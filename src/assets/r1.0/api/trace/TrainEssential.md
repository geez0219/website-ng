## TrainEssential
```python
TrainEssential(loss_keys:Set[str]) -> None
```
A trace to collect important information during training.    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

#### Args:

* **loss_keys** :  Which keys from the data dictionary correspond to loss values.    

### on_batch_begin
```python
on_batch_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs at the beginning of each batch.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        