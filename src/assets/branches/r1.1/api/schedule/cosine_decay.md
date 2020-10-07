## cosine_decay<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/schedule/lr_shedule.py/#L18-L61>View source on Github</a>
```python
cosine_decay(
	time: int,
	cycle_length: int,
	init_lr: float,
	min_lr: float=1e-06,
	start: int=1,
	cycle_multiplier: int=1
)
```
Learning rate cosine decay function (using half of cosine curve).

This method is useful for scheduling learning rates which oscillate over time:
```python
s = fe.schedule.LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
fe.Estimator(..., traces=[s])
```

For more information, check out SGDR: https://arxiv.org/pdf/1608.03983.pdf.


<h3>Args:</h3>

* **time** :  The current step or epoch during training starting from 1.
* **cycle_length** :  The decay cycle length.
* **init_lr** :  Initial learning rate to decay from.
* **min_lr** :  Minimum learning rate.
* **start** :  The step or epoch to start the decay schedule.
* **cycle_multiplier** :  The factor by which next cycle length will be multiplied.

<h3>Returns:</h3>

* **lr** :  learning rate given current step or epoch.

