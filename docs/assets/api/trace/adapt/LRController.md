## LRController
```python
LRController(model_name, lr_schedule=None, reduce_on_eval=False, reduce_patience=10, reduce_factor=0.1, reduce_mode='min', min_lr=1e-06)
```
Learning rate controller that makes learning rate follow the custom schedule and optionally reduces learningrate whenever evaluation loss meets certain condition.

#### Args:

* **model_name (str)** :  Model name of target model
* **lr_schedule (object, optional)** :  Scheduler that defines how learning rate changes. It should be `LRSchedule`        object. Defaults to None.
* **reduce_on_eval (bool, optional)** :  If true, it will reduce the learning rate when evaluation loss have been not        improving for several times. Defaults to False.
* **reduce_patience (int, optional)** :  Maximum accumulation of times of not being improving. Defaults to 10.
* **reduce_factor (float, optional)** :  Reduce factor of learning rate. Defaults to 0.1.
* **reduce_mode (str, optional)** :  It should be {"max", "min"}. If "max", the learning rate will reduce if        monitored number is too low. If "min", the learning rate will reduce if target is too high. Defaults to        "min".
* **min_lr (float, optional)** :  Minimum learning rate. Defaults to 1e-6.