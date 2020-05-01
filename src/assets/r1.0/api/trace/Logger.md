## Logger
```python
Logger(extra_log_keys:Set[str]) -> None
```
A Trace that prints log messages.    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

#### Args:

* **extra_log_keys** :  A set of keys to print from the system buffer besides those it would normally print.    

### on_batch_begin
```python
on_batch_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs at the beginning of each batch.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        

### on_epoch_begin
```python
on_epoch_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs at the beginning of each epoch.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        