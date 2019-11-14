## Estimator
```python
Estimator(pipeline, network, epochs, steps_per_epoch=None, validation_steps=None, traces=None, log_steps=100)
```
Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wrapsup `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other trainingnecessary information.

#### Args:

* **pipeline (obj)** :  Pipeline object that defines the data processing workflow. It should be an instance of        `fastestimator.pipepline.pipeline.Pipeline`
* **network (obj)** :  Network object that defines models and their external connection. It should be an instance of        `fastestimator.network.network.Network`
* **epochs (int)** :  Number of epooch to run.
* **steps_per_epoch ([type], optional)** :  Number of steps to run for each training session. If None, this will be the        training example number divided by batch_size. (round down). Defaults to None.
* **validation_steps ([type], optional)** :  Number of steps to run for each evaluation session, If None, this will be        the evaluation example number divided by batch_size (round down). Defaults to None.
* **traces (list, optional)** :  List of the traces objects to run during training. If None, there will be only basic        traces.
* **log_steps (int, optional)** :  Interval steps of logging. Defaults to 100.

### fit
```python
fit(self, summary=None)
```
Function to perform training on the estimator.

#### Args:

* **summary (str, optional)** :  Experiment name to return. If None, it won't return anything. Defaults to None.

#### Returns:
            Experiment object.        