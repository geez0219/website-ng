## ReduceLROnPlateau
```python
ReduceLROnPlateau(*args, **kwargs)
```
Reduce learning rate based on evaluation results.


#### Args:

* **model** :  A model instance compiled with fe.build.
* **metric** :  The metric name to be monitored. If None, the model's validation loss will be used as the metric.
* **patience** :  Number of epochs to wait before reducing LR again.
* **factor** :  Reduce factor for the learning rate.
* **best_mode** :  Higher is better ("max") or lower is better ("min").
* **min_lr** :  Minimum learning rate.

#### Raises:

* **AssertionError** :  If the loss cannot be inferred from the `model` and a `metric` was not provided.