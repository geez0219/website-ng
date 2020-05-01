## LRScheduler
```python
LRScheduler(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], lr_fn:Callable[[int], float]) -> None
```
Learning rate scheduler trace that changes the learning rate while training.
* **This class requires an input function which takes either 'epoch' or 'step' as input** :     ```python
* **s = LRScheduler(model=model, lr_fn=lambda step** :  fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))    fe.Estimator(..., traces=[s])  # Learning rate will change based on step
* **s = LRScheduler(model=model, lr_fn=lambda epoch** :  fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))    fe.Estimator(..., traces=[s])  # Learning rate will change based on epoch    ```

#### Args:

* **model** :  A model instance compiled with fe.build.
* **lr_fn** :  A lr scheduling function that takes either 'epoch' or 'step' as input.

#### Raises:

* **AssertionError** :  If the `lr_fn` is not configured properly.    