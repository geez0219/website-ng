## LRScheduler<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/adapt/lr_scheduler.py/#L32-L103>View source on Github</a>
```python
LRScheduler(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	lr_fn: Callable[[int], float]
)
-> None
```
Learning rate scheduler trace that changes the learning rate while training.

This class requires an input function which takes either 'epoch' or 'step' as input:
```python
s = LRScheduler(model=model, lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
fe.Estimator(..., traces=[s])  # Learning rate will change based on step
s = LRScheduler(model=model, lr_fn=lambda epoch: fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))
fe.Estimator(..., traces=[s])  # Learning rate will change based on epoch
```


<h3>Args:</h3>


* **model**: A model instance compiled with fe.build.

* **lr_fn**: A lr scheduling function that takes either 'epoch' or 'step' as input, or the string 'arc'. 

<h3>Raises:</h3>


* **AssertionError**: If the `lr_fn` is not configured properly.

