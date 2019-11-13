## TrainInfo
```python
TrainInfo()
```
Essential training information for logging during training. Please don't add this trace into an estimatormanually. An estimator will add it automatically.

#### Args:

* **log_steps (int)** :  Interval steps of logging

### on_batch_begin
```python
on_batch_begin(self, state)
```
Runs at the beginning of every batch of the mode.

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "mode" (str) :  current run time mode, can be "train", "eval" or "test"
 * "epoch" (int) :  current epoch index starting from 0
 * "train_step" (int) :  current global training step starting from 0
 * "batch_idx" (int) :  current local step of the epoch starting from 0
 * "batch_size" (int) :  current global batch size
 * "local_batch_size" (int) :  current batch size for single device                * any keys written by 'on_batch_begin' of previous traces        