## EvalEssential
```python
EvalEssential(loss_keys:Set[str], monitor_names:Set[str]) -> None
```
A trace to collect important information during evaluation.    Please don't add this trace into an estimator manually. FastEstimator will add it automatically.

#### Args:

* **loss_keys** :  Which keys from the data dictionary correspond to loss values.
* **monitor_names** :  Any other keys which should be collected over the course of an eval epoch.    

### on_batch_begin
```python
on_batch_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs at the beginning of each batch.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        

### on_begin
```python
on_begin(self, data:fastestimator.util.data.Data) -> None
```
Runs once at the beginning of training or testing.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        

### on_end
```python
on_end(self, data:fastestimator.util.data.Data) -> None
```
Runs once at the end training.

#### Args:

* **data** :  A dictionary through which traces can communicate with each other or write values for logging.        